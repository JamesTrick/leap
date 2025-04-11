from typing import Tuple, Sequence, Union
from dataclasses import dataclass
import torch.nn as nn
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class BaseLineConfig:
    vocab_size: int
    embed_dim: int = 128
    hidden_dim: int = 256
    pad_idx: int = None


class LeapBaselineModel(nn.Module):
    """Complete baseline model mainly to enable end-to-end testing of the training process.

    Comprised of an embedding layer, single LSTM layer, and two independent Linear layers. It is not really a 'Leap'
    model even as it doesn't predict the final token first.
    """
    def __init__(self,
                 cfg: BaseLineConfig,
        ):
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim
        self.padding_idx = cfg.pad_idx if cfg.pad_idx is not None else 0

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.fc_gap = nn.Linear(cfg.hidden_dim + cfg.hidden_dim, cfg.vocab_size)  # Predicts gap tokens
        self.fc_target = nn.Linear(cfg.hidden_dim, cfg.vocab_size)  # Predicts target token

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedded)  # lstm_out shape: [batch_size, seq_len, cfg.hidden_dim]
        final_hidden_state = lstm_out[:, -1, :].unsqueeze(1)

        target_output = self.fc_target(final_hidden_state.squeeze(1))  # Predicts final token

        target_state_expanded = final_hidden_state.expand(-1, lstm_out.size(1), -1)
        gap_input_features = torch.cat([lstm_out, target_state_expanded], dim=-1)
        gap_output = self.fc_gap(gap_input_features)

        return gap_output, target_output

    @torch.no_grad()
    def generate(self,
                 prompts: Union[str, Sequence[str]],
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 12
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.eval()

        results = list()

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', padding='max_length',  max_length=max_length, truncation=True)
            input_ids = inputs['input_ids']

            gap_logits, target_logits = self.forward(input_ids)

            target_pred_id = torch.argmax(target_logits, dim=-1)

            predicted_distant_token = tokenizer.decode(target_pred_id.cpu().numpy())

            gap_pred_ids = torch.argmax(gap_logits, dim=-1)
            predicted_gap_sequence = tokenizer.decode(gap_pred_ids[0], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "predicted_target_token": predicted_distant_token,
                "predicted_gap_sequence": predicted_gap_sequence,
            })
        return results

    @property
    def model_size(self) -> Tuple[int, float]:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_bytes = num_params * 4
        total_size_mb = total_bytes / (1024 * 1024)
        return num_params, total_size_mb


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config = {
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 128,
        'hidden_dim': 256,
        'pad_idx': tokenizer.pad_token_id,
    }

    model_config = BaseLineConfig(**config)
    model = LeapBaselineModel(model_config)
    prompt = "The quick brown fox"

    predictions = model.generate(prompt, tokenizer=tokenizer, max_length=5)

    print(predictions)
