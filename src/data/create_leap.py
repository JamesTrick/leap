import random
import torch
import ray

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@ray.remote
def process_sample(text: str,
                   tokenizer: PreTrainedTokenizerFast,
                   min_distance: int,
                   max_distance: int,
                   context_length: int,
                   mask_token_id: int):
    """Ray function to process each sample of given text from dataloader. It generates a random `gap_length` between
    `min_distance` and `max_distance` given a `target_token` to predict, and returns the `gap_tokens` in between.

    Args:
        text:
        tokenizer:
        min_distance:
        max_distance:
        context_length:
        mask_token_id:
    """
    tokenized_text = tokenizer.encode(text)

    if len(tokenized_text) < min_distance + 1:
        return None

    i = random.randint(0, max(0, len(tokenized_text) - min_distance - 1))

    if i + context_length > len(tokenized_text):
        return None

    context = tokenized_text[max(0, i - context_length + 1):i + 1]
    max_possible = min(max_distance, len(tokenized_text) - i - 1)
    if max_possible < min_distance:
        return None  # Skip if no valid gap

    distance = random.randint(min_distance, max_possible)
    gap_tokens = tokenized_text[i + 1:i + distance]

    # Ensure target token is a single token
    if i + distance < len(tokenized_text):
        target_token = tokenized_text[i + distance]
    else:
        # Use a default token if at the end
        target_token = tokenized_text[-1]

    masked_gap_tokens = [mask_token_id] * len(gap_tokens)

    return {
        "context": torch.tensor(context, dtype=torch.long),
        "masked_gap_tokens": torch.tensor(masked_gap_tokens, dtype=torch.long),
        "original_gap_tokens": torch.tensor(gap_tokens, dtype=torch.long),
        "target_token": torch.tensor([target_token], dtype=torch.long),
    }


class LeapDataset(Dataset):
    def __init__(self, texts, context_length=12, min_distance=2, max_distance=30, tokenizer="gpt2", mask_token_id=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.context_length = context_length

        if mask_token_id is None:
            if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
                self.mask_token_id = self.tokenizer.mask_token_id
            else:
                self.mask_token_id = 0
        else:
            self.mask_token_id = mask_token_id

        self.examples = []
        self.texts = texts

    def _get_distance(self, tokenized, start_idx):
        max_possible = min(self.max_distance, len(tokenized) - start_idx - 1)
        return random.randint(self.min_distance, max_possible) if max_possible >= self.min_distance else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        future = process_sample.remote(self.texts[idx], self.tokenizer, self.min_distance, self.max_distance,
                                       self.context_length, self.mask_token_id)
        sample = ray.get(future)

        if sample is None:
            return self.__getitem__(random.randint(0, len(self.texts) - 1))  # Retry with another sample
        return sample

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    contexts = [b["context"] for b in batch]
    masked_gaps = [b["masked_gap_tokens"] for b in batch]
    original_gaps = [b["original_gap_tokens"] for b in batch]
    target_tokens = [b["target_token"] for b in batch]

    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    padded_masked_gaps = pad_sequence(masked_gaps, batch_first=True, padding_value=0)
    padded_original_gaps = pad_sequence(original_gaps, batch_first=True, padding_value=0)

    batch_size = len(batch)

    if len(target_tokens) != batch_size:
        print(f"Warning: Target tokens count ({len(target_tokens)}) doesn't match batch size ({batch_size})")
        if len(target_tokens) > batch_size:
            # Take only the first element from each batch item
            target_tokens = [t[0:1] for t in target_tokens[:batch_size]]
        else:
            while len(target_tokens) < batch_size:
                target_tokens.append(torch.zeros(1, dtype=torch.long))

    # Stack target tokens ensuring they have the right shape [batch_size]
    stacked_targets = torch.stack([t.squeeze() for t in target_tokens])

    context_mask = (padded_contexts != 0).float()
    gap_mask = (padded_original_gaps != 0).float()

    return {
        "context": padded_contexts,
        "masked_gap_tokens": padded_masked_gaps,
        "original_gap_tokens": padded_original_gaps,
        "target_token": stacked_targets,
        "context_mask": context_mask,
        "gap_mask": gap_mask
    }
