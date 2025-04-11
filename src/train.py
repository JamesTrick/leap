import ray
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.models.baseline import LeapBaselineModel, BaseLineConfig
from src.data.create_leap import LeapDataset, collate_fn


def train_func(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        data,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    model_config = BaseLineConfig(
        **config['model_config'],
    )

    model = LeapBaselineModel(model_config).to(device)
    model = ray.train.torch.prepare_model(model)

    # Use ignore_index=0 for the loss to ignore padding tokens
    gap_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    target_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            optimizer.zero_grad()

            context = batch["context"].to(device)
            masked_gap_tokens = batch["masked_gap_tokens"].to(device)
            original_gap_tokens = batch["original_gap_tokens"].to(device)
            target_token = batch["target_token"].to(device)

            # Forward pass
            gap_preds, target_preds = model(context)

            # The critical issue is that gap_preds and original_gap_tokens must have compatible shapes
            # for the CrossEntropyLoss function:
            # - gap_preds needs shape [N, C] where N is batch_size*seq_len and C is vocab_size
            # - original_gap_tokens needs shape [N] with the same N

            # Create a mask for valid gap tokens (non-padding)
            valid_mask = (original_gap_tokens != 0).flatten()

            # Filter out only the valid predicted and target tokens
            # First reshape gap_preds to match the original_gap_tokens shape
            batch_size, context_seq_len, vocab_size = gap_preds.shape
            _, gap_seq_len = original_gap_tokens.shape

            # We need to be careful about sequence length
            # Only use the prediction for sequences we have targets for
            effective_seq_len = min(context_seq_len, gap_seq_len)

            # Reshape both tensors to 1D and 2D for loss calculation
            flattened_preds = gap_preds[:, :effective_seq_len, :].reshape(-1, vocab_size)
            flattened_targets = original_gap_tokens[:, :effective_seq_len].reshape(-1)

            # Apply the mask to get only valid positions
            if valid_mask.shape[0] > flattened_preds.shape[0]:
                valid_mask = valid_mask[:flattened_preds.shape[0]]
            elif valid_mask.shape[0] < flattened_preds.shape[0]:
                # Pad the mask if needed
                pad_size = flattened_preds.shape[0] - valid_mask.shape[0]
                padding = torch.zeros(pad_size, dtype=torch.bool, device=valid_mask.device)
                valid_mask = torch.cat([valid_mask, padding])

            # Now use the valid elements for loss calculation
            if valid_mask.sum() > 0:  # Only if we have valid elements
                filtered_preds = flattened_preds[valid_mask]
                filtered_targets = flattened_targets[valid_mask]

                print(f"Filtered gap preds shape: {filtered_preds.shape}")
                print(f"Filtered targets shape: {filtered_targets.shape}")

                # Now calculate gap loss with matching dimensions
                gap_loss = gap_loss_fn(filtered_preds, filtered_targets)
            else:
                # No valid elements, set loss to zero
                gap_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Ensure target token has the right shape
            if target_token.shape[0] != target_preds.shape[0]:
                print("Fixing target shape mismatch...")
                if target_token.shape[0] < target_preds.shape[0]:
                    # Repeat the last valid target
                    target_token = target_token.repeat(
                        (target_preds.shape[0] + target_token.shape[0] - 1) // target_token.shape[0])
                    target_token = target_token[:target_preds.shape[0]]
                else:
                    # Truncate
                    target_token = target_token[:target_preds.shape[0]]

            target_loss = target_loss_fn(target_preds, target_token)

            loss = gap_loss + target_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{config['epochs']}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        ray.train.report({"epoch": epoch, "train_loss": avg_loss})

    return model


if __name__ == "__main__":
    from datasets import load_dataset
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    dataset = load_dataset('stanfordnlp/imdb')
    data = LeapDataset([example['text'] for example in dataset['train']], tokenizer='gpt2')

    scaling_config = ScalingConfig(num_workers=2, use_gpu=torch.cuda.is_available())

    model_config = {
        'vocab_size': data.vocab_size,
        'embed_dim': 128,
        'hidden_dim': 256,
        'pad_idx': 0
    }

    trainer = TorchTrainer(
        train_func,
        train_loop_config={"epochs": 5, "batch_size": 16, "lr": 3e-4, "model_config": model_config},
        scaling_config=scaling_config
    )

    results = trainer.fit()
