import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Sequence, Dict
from dataclasses import dataclass


@dataclass
class LeapModelConfig:
    vocab_size: int
    embed_dim: int = 384
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    pad_idx: int = None
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    use_bidirectional: bool = True
    intermediate_size: int = 1024


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: LeapModelConfig):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.layer_norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None):
        attention_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask, need_weights=False)
        x = self.layer_norm1(x + attention_output)
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + ff_output)
        return x


class LeapModel(nn.Module):
    def __init__(self, cfg: LeapModelConfig):
        super().__init__()
        self.config = cfg
        self.vocab_size = cfg.vocab_size
        self.embed_dim = cfg.embed_dim
        self.hidden_dim = cfg.hidden_dim
        self.padding_idx = cfg.pad_idx

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, self.embed_dim)

        self.input_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(cfg) for _ in range(cfg.num_layers)
        ])

        self.target_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.gap_classifier = nn.Linear(self.hidden_dim * 2, self.vocab_size)
        self.target_classifier = nn.Linear(self.hidden_dim, self.vocab_size)

        self.target_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps),
            nn.GELU()
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.padding_idx].fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_attention_mask(self, input_ids):
        return input_ids == self.padding_idx

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)

        input_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        hidden_states = input_embeddings + position_embeddings
        hidden_states = self.input_proj(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        q = torch.mean(hidden_states, dim=1, keepdim=True)
        context_vector, v = self.target_attention(q, hidden_states, hidden_states)
        context_vector = context_vector.squeeze(1)

        target_features = self.target_proj(context_vector)
        target_logits = self.target_classifier(target_features)

        expanded_target = target_features.unsqueeze(1).expand(-1, seq_len, -1)

        gap_features = torch.cat([hidden_states, expanded_target], dim=-1)
        gap_logits = self.gap_classifier(gap_features)

        return {
            'gap_logits': gap_logits,
            'target_logits': target_logits,
            'hidden_states': hidden_states,
            'context_vector': context_vector
        }

    def generate(self,
                 prompts: Union[str, Sequence[str]],
                 tokenizer,
                 max_length: int = 512,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 temperature: float = 0.7,
                 ) -> Dict:
        if isinstance(prompts, str):
            prompts = [prompts]

        self.eval()
        results = []

        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                max_length=max_length,
                truncation=True
            )

            input_ids = encoded['input_ids'].to(next(self.parameters()).device)
            attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)

            # Forward pass
            outputs = self.forward(input_ids, attention_mask=(attention_mask == 0))

            gap_logits = outputs['gap_logits']
            target_logits = outputs['target_logits']

            # Apply temperature
            gap_logits = gap_logits / temperature
            target_logits = target_logits / temperature

            target_probs = F.softmax(target_logits, dim=-1)

            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(target_probs, k=top_k, dim=-1)

            # Top-p filtering (nucleus sampling)
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Get indices of tokens to keep for each sample
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            filtered_probs = top_k_probs.masked_fill(indices_to_remove, 0.0)

            # Sample from the filtered distribution
            if filtered_probs.sum() == 0:
                target_pred_id = torch.argmax(target_logits, dim=-1)
            else:
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                target_pred_id = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
                target_pred_id = torch.gather(
                    top_k_indices, -1, target_pred_id.unsqueeze(-1)
                ).squeeze(-1)

            # Simple greedy decoding for gap tokens
            gap_pred_ids = torch.argmax(gap_logits, dim=-1)

            target_token = tokenizer.decode(target_pred_id.cpu().numpy())
            gap_sequence = tokenizer.decode(gap_pred_ids[0], skip_special_tokens=True)

            result = {
                "prompt": prompt,
                "predicted_target_token": target_token,
                "predicted_gap_sequence": gap_sequence,
            }

            results.append(result)
        return results
