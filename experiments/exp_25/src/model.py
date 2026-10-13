"""
exp_25: LOB Vision Transformer (LOB-ViT) for Regression

Architecture based on LiT paper - treating LOB as 2D image-like structure:
- Input: Raw 32 features reshaped to (H=12 levels, W=T timesteps, C=3 channels)
  - H: 6 bid levels + 6 ask levels = 12
  - W: Temporal window (configurable, e.g., 10)
  - C: 3 channels (Price, Volume, Delta/auxiliary)

- 2D Patch Embedding: Split spatial grid into patches, linear project
- Position Embedding: Learnable 2D positional encoding
- Transformer Encoder: Self-attention on patch sequence (causal temporal mask)
- Temporal Pooling: Pool across patches per timestep
- GRU: Final temporal modeling across timesteps
- Regression Head: Linear output for t0, t1

Key insight from paper: LOB has spatial structure (bid/ask levels) that can be 
captured with 2D convolutions/attention, similar to image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LOBReshaper(nn.Module):
    """
    Reshape raw 32 LOB features into 2D spatial format (H, W, C).
    
    Raw features layout (32 total):
    - p0-p5: bid prices (6)      -> rows 0-5, channel 0 (price)
    - p6-p11: ask prices (6)     -> rows 6-11, channel 0 (price)
    - v0-v5: bid volumes (6)     -> rows 0-5, channel 1 (volume)
    - v6-v11: ask volumes (6)    -> rows 6-11, channel 1 (volume)
    - dp0-dp3: delta prices (4)  -> rows 0-3, channel 2 (delta) - padded
    - dv0-dv3: delta volumes (4) -> rows 4-7, channel 2 (delta) - padded
    
    Output: (batch, seq_len, H=12, C=3) per timestep
    When stacking W timesteps: (batch, H=12, W=timesteps, C=3)
    """
    
    def __init__(self, num_levels=12, num_channels=3):
        super().__init__()
        self.num_levels = num_levels  # 6 bid + 6 ask
        self.num_channels = num_channels  # price, volume, delta
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 32) raw LOB features
        Returns:
            reshaped: (batch, seq_len, H=12, C=3)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize output tensor
        out = torch.zeros(batch, seq_len, self.num_levels, self.num_channels, 
                         device=x.device, dtype=x.dtype)
        
        # Channel 0: Prices
        # Bid prices p0-p5 -> rows 0-5
        out[:, :, 0:6, 0] = x[:, :, 0:6]
        # Ask prices p6-p11 -> rows 6-11
        out[:, :, 6:12, 0] = x[:, :, 6:12]
        
        # Channel 1: Volumes
        # Bid volumes v0-v5 -> rows 0-5
        out[:, :, 0:6, 1] = x[:, :, 12:18]
        # Ask volumes v6-v11 -> rows 6-11
        out[:, :, 6:12, 1] = x[:, :, 18:24]
        
        # Channel 2: Deltas (only 8 features, pad the rest)
        # Delta prices dp0-dp3 -> rows 0-3 
        out[:, :, 0:4, 2] = x[:, :, 24:28]
        # Delta volumes dv0-dv3 -> rows 4-7
        out[:, :, 4:8, 2] = x[:, :, 28:32]
        # Rows 8-11 remain zero (padding)
        # Try to keep deltas in top half of levels to allow some spatial locality for attention
        return out


class LOB2DPatchEmbedding(nn.Module):
    """
    2D Patch Embedding for LOB data arranged as (H, W, C).
    
    Similar to ViT: splits the 2D grid into patches and projects to embedding dim.
    Patches are created along both height (levels) and width (time) dimensions.
    
    For LOB: patch_size_h controls how many levels per patch,
             patch_size_w controls how many timesteps per patch.
    """
    
    def __init__(self, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.embed_dim = embed_dim
        
        # Number of patches in each dimension
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch dimension: patch_h * patch_w * channels
        patch_dim = patch_size_h * patch_size_w * num_channels
        
        # Linear projection
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, H=12, W=window_size, C=3)
        Returns:
            patches: (batch, num_patches, embed_dim)
            patch_positions: tensor of (h_idx, w_idx) for each patch
        """
        batch, H, W, C = x.shape
        
        # Ensure dimensions are compatible
        assert H == self.num_levels, f"Height {H} != num_levels {self.num_levels}"
        
        # Handle variable window sizes by padding if needed
        if W < self.window_size:
            pad_w = self.window_size - W
            x = F.pad(x, (0, 0, pad_w, 0))  # Pad at beginning (causal)
            W = self.window_size
        elif W > self.window_size:
            x = x[:, :, -self.window_size:, :]  # Take last window_size
            W = self.window_size
        
        # Reshape to patches: (batch, num_patches_h, patch_h, num_patches_w, patch_w, C)
        x = x.view(batch, self.num_patches_h, self.patch_size_h,
                   self.num_patches_w, self.patch_size_w, C)
        
        # Reorder: (batch, num_patches_h, num_patches_w, patch_h, patch_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Flatten patches: (batch, num_patches, patch_dim)
        x = x.view(batch, self.num_patches, -1)
        
        # Project to embedding dimension
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class LearnablePositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for patches.
    Separate embeddings for height and width positions.
    """
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Separate embeddings for height and width
        self.pos_embed_h = nn.Embedding(num_patches_h, embed_dim // 2)
        self.pos_embed_w = nn.Embedding(num_patches_w, embed_dim // 2)
        
        # Alternative: single embedding for all positions
        self.pos_embed = nn.Embedding(num_patches_h * num_patches_w, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            x + positional encoding
        """
        batch, num_patches, _ = x.shape
        positions = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch, -1)
        return x + self.pos_embed(positions)


class CausalTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional causal masking.
    For 2D patches, we apply causal mask only along the temporal (W) dimension.
    """
    
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, dropout=0.1):
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
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class CausalTransformerEncoder(nn.Module):
    """Stack of transformer encoder layers with causal masking."""
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1,
                 num_patches_h=3, num_patches_w=5):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask_2d(self, num_patches, device):
        """
        Generate causal mask for 2D patches.
        Patches are ordered row-major: (h0,w0), (h0,w1), ..., (h1,w0), (h1,w1), ...
        We allow attention to all patches in current and previous time columns.
        """
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        
        mask = torch.zeros(num_patches, num_patches, device=device)
        
        for i in range(num_patches):
            i_w = i % num_w  # Temporal position of patch i
            for j in range(num_patches):
                j_w = j % num_w  # Temporal position of patch j
                if j_w > i_w:
                    # Can't attend to future time columns
                    mask[i, j] = float('-inf')
        
        return mask
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            x: (batch, num_patches, embed_dim)
        """
        num_patches = x.shape[1]
        attn_mask = self.generate_causal_mask_2d(num_patches, x.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x


class TemporalPooling(nn.Module):
    """
    Pool patch embeddings along the level dimension to get per-timestep features.
    """
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim, pool_type='mean'):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, embed_dim) where patches are (h, w) ordered row-major
        Returns:
            pooled: (batch, num_patches_w, embed_dim) - one feature per timestep
        """
        batch, num_patches, embed_dim = x.shape
        
        # Reshape to (batch, num_patches_h, num_patches_w, embed_dim)
        x = x.view(batch, self.num_patches_h, self.num_patches_w, embed_dim)
        
        if self.pool_type == 'mean':
            # Mean pool across height dimension
            pooled = x.mean(dim=1)  # (batch, num_patches_w, embed_dim)
        elif self.pool_type == 'max':
            pooled = x.max(dim=1)[0]
        elif self.pool_type == 'attention':
            # Attention-weighted pooling
            attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)  # (batch, num_h, num_w)
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, num_w, embed_dim)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return pooled


class LOBViTModel(nn.Module):
    """
    LOB Vision Transformer (LOB-ViT) for regression.
    
    Architecture:
    1. Reshape raw features to 2D (H=levels, W=time, C=channels)
    2. 2D Patch Embedding (like ViT)
    3. Positional Encoding (2D)
    4. Causal Transformer Encoder
    5. Temporal Pooling (across level patches)
    6. GRU (temporal modeling)
    7. Regression Head
    """
    
    def __init__(self, input_dim=32, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64,
                 num_transformer_layers=2, num_heads=4, ff_dim=256,
                 gru_hidden=128, gru_layers=2, output_dim=2, dropout=0.1,
                 pool_type='mean'):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Calculate patch grid dimensions
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 1. Reshape raw features to 2D
        self.reshaper = LOBReshaper(num_levels, num_channels)
        
        # 2. 2D Patch Embedding
        self.patch_embed = LOB2DPatchEmbedding(
            num_levels, num_channels, window_size,
            patch_size_h, patch_size_w, embed_dim
        )
        
        # 3. Positional Encoding
        self.pos_encoding = LearnablePositionalEncoding2D(
            self.num_patches_h, self.num_patches_w, embed_dim
        )
        
        # 4. Transformer Encoder
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout,
            self.num_patches_h, self.num_patches_w
        )
        
        # 5. Temporal Pooling
        self.temporal_pool = TemporalPooling(
            self.num_patches_h, self.num_patches_w, embed_dim, pool_type
        )
        
        # 6. GRU
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # 7. Output Head
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, 32) raw LOB features
            hidden: GRU hidden state
        Returns:
            predictions: (batch, seq_len, 2)
            hidden: updated GRU hidden state
        """
        batch, seq_len, _ = x.shape
        
        # Reshape to 2D: (batch, seq_len, H=12, C=3)
        x_2d = self.reshaper(x)
        
        # Process with sliding window
        outputs = []
        
        for t in range(seq_len):
            # Get window ending at current timestep
            start_t = max(0, t - self.window_size + 1)
            window = x_2d[:, start_t:t+1, :, :]  # (batch, window_len, H, C)
            
            # Pad if window is smaller than window_size
            window_len = window.shape[1]
            if window_len < self.window_size:
                pad_len = self.window_size - window_len
                padding = torch.zeros(batch, pad_len, self.num_levels, self.num_channels,
                                     device=x.device, dtype=x.dtype)
                window = torch.cat([padding, window], dim=1)
            
            # Rearrange to (batch, H, W, C) - swap seq and levels dims
            window = window.permute(0, 2, 1, 3)  # (batch, H=12, W=window_size, C=3)
            
            # 2D Patch embedding
            patches = self.patch_embed(window)  # (batch, num_patches, embed_dim)
            
            # Add positional encoding
            patches = self.pos_encoding(patches)
            
            # Transformer
            transformer_out = self.transformer(patches)  # (batch, num_patches, embed_dim)
            
            # Temporal pooling - get last timestep
            pooled = self.temporal_pool(transformer_out)  # (batch, num_patches_w, embed_dim)
            pooled = pooled[:, -1:, :]  # Take last time position (batch, 1, embed_dim)
            
            # GRU step
            gru_out, hidden = self.gru(pooled, hidden)
            
            # Output
            out = self.output_head(gru_out.squeeze(1))  # (batch, 2)
            outputs.append(out)
        
        predictions = torch.stack(outputs, dim=1)  # (batch, seq_len, 2)
        
        return predictions, hidden
    
    def forward_full_sequence(self, x, hidden=None):
        """
        Optimized forward for training: process full sequence at once.
        Still maintains causality through masking.
        
        Args:
            x: (batch, seq_len, 32)
        Returns:
            predictions: (batch, seq_len, 2)
            hidden: final GRU hidden state
        """
        batch, seq_len, _ = x.shape
        
        # Reshape to 2D: (batch, seq_len, H=12, C=3)
        x_2d = self.reshaper(x)
        
        # For efficient training, we'll process in chunks
        # Each chunk uses a sliding window approach
        
        # Pad sequence at beginning for causal windowing
        pad_len = self.window_size - 1
        padding = torch.zeros(batch, pad_len, self.num_levels, self.num_channels,
                             device=x.device, dtype=x.dtype)
        x_2d_padded = torch.cat([padding, x_2d], dim=1)  # (batch, seq_len + pad_len, H, C)
        
        # Create all windows using unfold
        # (batch, seq_len, window_size, H, C)
        windows = x_2d_padded.unfold(1, self.window_size, 1)  # (batch, seq_len, H, C, window_size)
        windows = windows.permute(0, 1, 4, 2, 3).contiguous()  # (batch, seq_len, window_size, H, C)
        
        # Reshape for batch processing
        # (batch * seq_len, window_size, H, C)
        batch_windows = windows.reshape(batch * seq_len, self.window_size, self.num_levels, self.num_channels)
        
        # Rearrange to (batch * seq_len, H, W, C)
        batch_windows = batch_windows.permute(0, 2, 1, 3)
        
        # 2D Patch embedding
        patches = self.patch_embed(batch_windows)  # (batch * seq_len, num_patches, embed_dim)
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Transformer
        transformer_out = self.transformer(patches)  # (batch * seq_len, num_patches, embed_dim)
        
        # Temporal pooling
        pooled = self.temporal_pool(transformer_out)  # (batch * seq_len, num_patches_w, embed_dim)
        pooled = pooled[:, -1, :]  # Take last time position (batch * seq_len, embed_dim)
        
        # Reshape back to sequence
        pooled = pooled.view(batch, seq_len, self.embed_dim)
        
        # GRU over full sequence
        gru_out, hidden = self.gru(pooled, hidden)  # (batch, seq_len, gru_hidden)
        
        # Output head
        predictions = self.output_head(gru_out)  # (batch, seq_len, 2)
        
        return predictions, hidden


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing LOB-ViT Model...")
    
    # Configuration
    config = {
        'input_dim': 32,
        'num_levels': 12,
        'num_channels': 3,
        'window_size': 10,
        'patch_size_h': 4,  # 12 / 4 = 3 patches in height
        'patch_size_w': 2,  # 10 / 2 = 5 patches in width
        'embed_dim': 64,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'ff_dim': 256,
        'gru_hidden': 128,
        'gru_layers': 2,
        'dropout': 0.1,
        'pool_type': 'mean',
    }
    
    model = LOBViTModel(**config)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test full sequence forward (training mode)
    model.train()
    output, hidden = model.forward_full_sequence(x)
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test single-step forward (inference mode)
    model.eval()
    with torch.no_grad():
        output_step, hidden_step = model(x[:, :20, :])
        print(f"Single-step output shape: {output_step.shape}")
    
    print("\nModel architecture:")
    print(model)
