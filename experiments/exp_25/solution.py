"""
exp_25: LOB Vision Transformer - Submission Solution

Fast incremental inference for LOB-ViT model.
Uses sliding window of recent timesteps, reshapes to 2D, processes through
patch embedding + transformer + GRU.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DataPoint


class LOBReshaper(nn.Module):
    """Reshape raw 32 LOB features into 2D spatial format (H, W, C)."""
    
    def __init__(self, num_levels=12, num_channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = num_channels
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 32) raw LOB features
        Returns:
            reshaped: (batch, seq_len, H=12, C=3)
        """
        batch, seq_len, _ = x.shape
        out = torch.zeros(batch, seq_len, self.num_levels, self.num_channels,
                         device=x.device, dtype=x.dtype)
        
        # Channel 0: Prices (bid p0-p5 -> rows 0-5, ask p6-p11 -> rows 6-11)
        out[:, :, 0:6, 0] = x[:, :, 0:6]
        out[:, :, 6:12, 0] = x[:, :, 6:12]
        
        # Channel 1: Volumes (bid v0-v5 -> rows 0-5, ask v6-v11 -> rows 6-11)
        out[:, :, 0:6, 1] = x[:, :, 12:18]
        out[:, :, 6:12, 1] = x[:, :, 18:24]
        
        # Channel 2: Deltas (dp0-dp3 -> rows 0-3, dv0-dv3 -> rows 4-7)
        out[:, :, 0:4, 2] = x[:, :, 24:28]
        out[:, :, 4:8, 2] = x[:, :, 28:32]
        
        return out


class LOB2DPatchEmbedding(nn.Module):
    """2D Patch Embedding for LOB data."""
    
    def __init__(self, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.embed_dim = embed_dim
        
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        patch_dim = patch_size_h * patch_size_w * num_channels
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, H=12, W=window_size, C=3)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        batch, H, W, C = x.shape
        
        if W < self.window_size:
            pad_w = self.window_size - W
            x = F.pad(x, (0, 0, pad_w, 0))
            W = self.window_size
        elif W > self.window_size:
            x = x[:, :, -self.window_size:, :]
            W = self.window_size
        
        x = x.view(batch, self.num_patches_h, self.patch_size_h,
                   self.num_patches_w, self.patch_size_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, self.num_patches, -1)
        
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class LearnablePositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding."""
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim):
        super().__init__()
        self.pos_embed = nn.Embedding(num_patches_h * num_patches_w, embed_dim)
    
    def forward(self, x):
        batch, num_patches, _ = x.shape
        positions = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch, -1)
        return x + self.pos_embed(positions)


class CausalTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with causal masking."""
    
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
    """Stack of transformer encoder layers with causal masking."""
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.0,
                 num_patches_h=3, num_patches_w=5):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask_2d(self, num_patches, device):
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        
        mask = torch.zeros(num_patches, num_patches, device=device)
        for i in range(num_patches):
            i_w = i % num_w
            for j in range(num_patches):
                j_w = j % num_w
                if j_w > i_w:
                    mask[i, j] = float('-inf')
        return mask
    
    def forward(self, x):
        num_patches = x.shape[1]
        attn_mask = self.generate_causal_mask_2d(num_patches, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class TemporalPooling(nn.Module):
    """Pool patch embeddings along level dimension."""
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim, pool_type='mean'):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            self.attention = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        batch, num_patches, embed_dim = x.shape
        x = x.view(batch, self.num_patches_h, self.num_patches_w, embed_dim)
        
        if self.pool_type == 'mean':
            pooled = x.mean(dim=1)
        elif self.pool_type == 'max':
            pooled = x.max(dim=1)[0]
        elif self.pool_type == 'attention':
            attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return pooled


class LOBViTModel(nn.Module):
    """LOB Vision Transformer for regression."""
    
    def __init__(self, input_dim=32, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64,
                 num_transformer_layers=2, num_heads=4, ff_dim=256,
                 gru_hidden=128, gru_layers=2, output_dim=2, dropout=0.0,
                 pool_type='mean'):
        super().__init__()
        
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        
        self.reshaper = LOBReshaper(num_levels, num_channels)
        self.patch_embed = LOB2DPatchEmbedding(
            num_levels, num_channels, window_size, patch_size_h, patch_size_w, embed_dim
        )
        self.pos_encoding = LearnablePositionalEncoding2D(
            self.num_patches_h, self.num_patches_w, embed_dim
        )
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout,
            self.num_patches_h, self.num_patches_w
        )
        self.temporal_pool = TemporalPooling(
            self.num_patches_h, self.num_patches_w, embed_dim, pool_type
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
    
    def forward_single_window(self, window_2d, hidden=None):
        """
        Process a single window for incremental inference.
        
        Args:
            window_2d: (batch, H=12, W=window_size, C=3)
            hidden: GRU hidden state
        Returns:
            predictions: (batch, 2)
            hidden: updated hidden state
        """
        # Patch embedding
        patches = self.patch_embed(window_2d)
        patches = self.pos_encoding(patches)
        
        # Transformer
        transformer_out = self.transformer(patches)
        
        # Pool to get per-timestep features, take last
        pooled = self.temporal_pool(transformer_out)
        pooled = pooled[:, -1:, :]  # (batch, 1, embed_dim)
        
        # GRU step
        gru_out, hidden = self.gru(pooled, hidden)
        
        # Output
        predictions = self.output_head(gru_out.squeeze(1))
        
        return predictions, hidden


class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        self.feature_buffer = None  # Store recent raw features
        self.window_size = 10
        
        self.running_mean = None
        self.running_var = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        self.device = torch.device("cpu")
        self.model = LOBViTModel(
            input_dim=32, num_levels=12, num_channels=3, window_size=10,
            patch_size_h=4, patch_size_w=2, embed_dim=64,
            num_transformer_layers=2, num_heads=4, ff_dim=256,
            gru_hidden=128, gru_layers=2, dropout=0.0, pool_type='mean'
        )
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded LOB-ViT model from {model_path}")
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
            self.feature_buffer = None
            self.running_mean = None
            self.running_var = None
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Normalize and add to buffer
        normalized_state = self._normalize_features(data_point.state)
        
        if self.feature_buffer is None:
            self.feature_buffer = normalized_state.reshape(1, 32)
        else:
            self.feature_buffer = np.vstack([self.feature_buffer, normalized_state.reshape(1, 32)])
            if self.feature_buffer.shape[0] > self.window_size:
                self.feature_buffer = self.feature_buffer[-self.window_size:]
        
        # Create window tensor
        window_len = self.feature_buffer.shape[0]
        if window_len < self.window_size:
            padding = np.zeros((self.window_size - window_len, 32))
            window = np.vstack([padding, self.feature_buffer])
        else:
            window = self.feature_buffer
        
        # Reshape to (1, seq_len, 32)
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        
        # Reshape to 2D: (1, H=12, W=window_size, C=3)
        window_2d = self.model.reshaper(window_tensor)
        window_2d = window_2d.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # (1, H, W, C)
        
        with torch.no_grad():
            output, self.hidden = self.model.forward_single_window(window_2d, self.hidden)
        
        if not data_point.need_prediction:
            return None
        
        prediction = output.numpy().flatten()
        return np.clip(prediction, -6.0, 6.0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from src.utils import ScorerStepByStep
    
    print("Testing exp_25 LOB-ViT model...")
    model = PredictionModel()
    scorer = ScorerStepByStep("datasets/valid.parquet")
    results = scorer.score(model)
    
    print(f"\nResults:")
    print(results)