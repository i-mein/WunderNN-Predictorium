"""
exp_24: LiT-inspired LOB Transformer - Fast Submission Solution
Uses incremental single-step inference with GRU hidden state.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from utils import DataPoint


class LOBPatchEmbedding(nn.Module):
    def __init__(self, input_dim=32, embed_dim=64, patch_size=1):
        super().__init__()
        self.projection = nn.Linear(input_dim * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x, start_pos=0):
        batch, seq_len, _ = x.shape
        # Use modular positions to stay within bounds
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device) % self.max_len
        positions = positions.unsqueeze(0).expand(batch, -1)
        return x + self.pos_embedding(positions)


class CausalTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CausalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        seq_len = x.shape[1]
        attn_mask = self.generate_causal_mask(seq_len, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class LiTModel(nn.Module):
    def __init__(self, input_dim=32, embed_dim=64, patch_size=1, num_transformer_layers=2,
                 num_heads=4, ff_dim=256, gru_hidden=128, gru_layers=2, output_dim=2,
                 dropout=0.0, max_seq_len=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = LOBPatchEmbedding(input_dim, embed_dim, patch_size)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout
        )
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward_single_step(self, x, hidden=None, context_buffer=None, step_idx=0):
        """Single-step forward for fast inference."""
        # Embed current input
        x_embed = self.patch_embed(x)  # (batch, 1, embed_dim)
        
        # Build context for transformer
        if context_buffer is not None:
            full_context = torch.cat([context_buffer, x_embed], dim=1)
        else:
            full_context = x_embed
        
        # Add positional encoding (relative to context window, not absolute step)
        context_len = full_context.shape[1]
        full_context = self.pos_encoding(full_context, start_pos=0)
        
        # Transformer
        transformer_out = self.transformer(full_context)
        current_out = transformer_out[:, -1:, :]
        
        # GRU step
        gru_out, hidden = self.gru(current_out, hidden)
        
        # Output
        predictions = self.output_head(gru_out.squeeze(1))
        
        # Update context buffer
        new_context = x_embed
        
        return predictions, hidden, new_context


class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        self.context_buffer = None
        self.step_idx = 0
        self.context_len = 20
        
        self.running_mean = None
        self.running_var = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        self.device = torch.device("cpu")
        self.model = LiTModel(
            input_dim=32, embed_dim=64, patch_size=1,
            num_transformer_layers=2, num_heads=4, ff_dim=256,
            gru_hidden=128, gru_layers=2, dropout=0.0, max_seq_len=100
        )
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded LiT model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _normalize_features(self, state):
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        if self.running_mean is None:
            self.running_mean = state.copy()
            self.running_var = np.ones_like(state)
            return np.zeros_like(state)
        alpha = 0.01
        self.running_mean = (1 - alpha) * self.running_mean + alpha * state
        self.running_var = (1 - alpha) * self.running_var + alpha * (state - self.running_mean) ** 2
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (state - self.running_mean) / std
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
            self.context_buffer = None
            self.step_idx = 0
            self.running_mean = None
            self.running_var = None
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        normalized_state = self._normalize_features(data_point.state)
        input_tensor = torch.tensor(normalized_state, dtype=torch.float32).view(1, 1, 32)
        
        with torch.no_grad():
            output, self.hidden, new_embed = self.model.forward_single_step(
                input_tensor, self.hidden, self.context_buffer, self.step_idx
            )
        
        # Update context buffer (sliding window)
        if self.context_buffer is None:
            self.context_buffer = new_embed
        else:
            self.context_buffer = torch.cat([self.context_buffer, new_embed], dim=1)
            if self.context_buffer.shape[1] > self.context_len:
                self.context_buffer = self.context_buffer[:, -self.context_len:, :]
        
        self.step_idx += 1
        
        if not data_point.need_prediction:
            return None
        
        prediction = output.numpy().flatten()
        return np.clip(prediction, -6.0, 6.0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from src.utils import ScorerStepByStep
    
    print("Testing exp_24 LiT model (fast version)...")
    model = PredictionModel()
    scorer = ScorerStepByStep("datasets/valid.parquet")
    results = scorer.score(model)
    
    print(f"\nResults:")
    print(results)
    # print(f"  t0 correlation: {results['t0_corr']:.4f}")
    # print(f"  t1 correlation: {results['t1_corr']:.4f}")
    # print(f"  Weighted score: {results['weighted_score']:.4f}")
