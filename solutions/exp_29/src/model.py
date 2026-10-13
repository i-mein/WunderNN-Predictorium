"""
exp_29: LOBViT-style 2D Transformer Model

Architecture:
1. Reshape raw 32 features to 2D: (B, 12, T, C)
   - 12 LOB levels (6 bid + 6 ask)
   - T timesteps
   - C channels (price, volume, optional delta)
   
2. Patch embedding: Split into (patch_h, patch_w) patches
   - Each patch becomes a token
   - Linear projection to embed_dim
   
3. Positional encoding (learnable 2D)

4. Transformer encoder with:
   - Multi-head self-attention
   - MLP blocks
   - Stochastic depth (drop_path)
   
5. CLS token pooling → Regression head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth - drop paths (residual blocks) during training."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed2D(nn.Module):
    """
    2D Patch Embedding for LOB data.
    
    Input: (B, H, W, C) where H=levels, W=timesteps, C=channels
    Output: (B, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        num_levels: int = 12,
        context_len: int = 100,
        num_channels: int = 3,
        patch_size: tuple = (3, 10),
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.context_len = context_len
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_h = num_levels // patch_size[0]  # 12 // 3 = 4
        self.num_patches_w = context_len // patch_size[1]  # 100 // 10 = 10
        self.num_patches = self.num_patches_h * self.num_patches_w  # 40
        
        # Patch embedding: Conv2d with kernel=stride=patch_size
        self.proj = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) = (B, num_levels, context_len, num_channels)
        Returns:
            (B, num_patches, embed_dim)
        """
        # Permute to (B, C, H, W) for Conv2d
        x = x.permute(0, 3, 1, 2)  # (B, C, 12, 100)
        
        # Patch projection
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        x = self.norm(x)
        
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class LOBViT(nn.Module):
    """
    LOB Vision Transformer for price prediction.
    
    Treats LOB data as a 2D image:
    - Height: LOB levels (bid/ask prices at different depths)
    - Width: Time steps
    - Channels: Price, Volume, Delta
    """
    
    def __init__(
        self,
        num_levels: int = 12,
        context_len: int = 100,
        num_channels: int = 2, #3, --- IGNORE ---
        patch_size: tuple = (3, 10),
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        output_dim: int = 2,
        use_cls_token: bool = True,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        # add optional delta channel feats
        use_delta_feats: bool = False, #True,
        vb: bool = False,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.context_len = context_len
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.vb = vb
        self.use_delta_feats = use_delta_feats
        if self.use_delta_feats:
            self.num_channels += 1  # Add delta channel if enabled

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            num_levels=num_levels,
            context_len=context_len,
            num_channels=num_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embedding (learnable)
        num_pos = num_patches + 1 if use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks with increasing drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, num_levels, context_len, num_channels) = (B, 12, 100, 3)
        Returns:
            (B, output_dim) predictions
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        if self.vb: print(f"Patch emb out shape: {x.shape}")
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if self.vb: print(f"CLS token added shape: {x.shape}")
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.vb: print(f"After pos emb shape: {x.shape}")
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        if self.vb: print(f"After Transformer blocks shape: {x.shape}")
        
        # Pool: use CLS token or mean
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        if self.vb: print(f"After pooling shape: {x.shape}")

        # Regression head
        out = self.head(x)
        if self.vb: print(f"Output shape: {out.shape}")
        
        return out


def reshape_raw_to_2d(raw_features: torch.Tensor, context_len: int,
                      use_delta_feats: bool = False) -> torch.Tensor:
    """
    Reshape raw 32 LOB features to 2D format.
    
    Raw features (32):
    - p0-p5: bid prices (levels 0-5)
    - p6-p11: ask prices (levels 6-11)
    - v0-v5: bid volumes (levels 0-5)
    - v6-v11: ask volumes (levels 6-11)
    - dp0-dp3: delta prices (4 values)
    - dv0-dv3: delta volumes (4 values)
    
    Args:
        raw_features: (B, T, 32) raw LOB features
        context_len: Expected context length
        
    Returns:
        (B, 12, T, 3) reshaped features
        - Dim 1: 12 levels (6 bid + 6 ask)
        - Dim 2: T timesteps
        - Dim 3: 3 channels (price, volume, temporal_delta)
    """
    B, T, num_features = raw_features.shape
    assert num_features == 32, f"Expected 32 features, got {num_features}"
    
    # Extract price and volume for each level
    # Prices: p0-p5 (bid), p6-p11 (ask)
    prices = torch.cat([
        raw_features[:, :, 0:6],   # Bid prices
        raw_features[:, :, 6:12],  # Ask prices
    ], dim=2)  # (B, T, 12)
    
    # Volumes: v0-v5 (bid), v6-v11 (ask)
    volumes = torch.cat([
        raw_features[:, :, 12:18],  # Bid volumes
        raw_features[:, :, 18:24],  # Ask volumes
    ], dim=2)  # (B, T, 12)
    
    # Compute temporal delta (price change over time) - CAUSAL
    if use_delta_feats:
        # Use padded diff: delta[t] = price[t] - price[t-1]
        prices_padded = F.pad(prices, (0, 0, 1, 0))[:, :-1, :]  # Shift right
        temporal_delta = prices - prices_padded  # (B, T, 12)
        # First timestep delta is 0 (no previous value)
        temporal_delta[:, 0, :] = 0
    
        # Stack channels: (B, T, 12, 3)
        x = torch.stack([prices, volumes, temporal_delta], dim=3)

    else:
        # Stack channels without delta: (B, T, 12, 2)
        x = torch.stack([prices, volumes], dim=3)

    # Permute to (B, 12, T, F) for patch embedding
    x = x.permute(0, 2, 1, 3)  # (B, levels, timesteps, channels)

    return x
