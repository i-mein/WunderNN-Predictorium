"""
exp_23b: Multi-Stream TCN + GRU with Attention

Architecture:
- Separate TCN streams for each feature category (prices, volumes, deltas, depth)
- Feature fusion with attention mechanism
- GRU for temporal modeling
- Separate regression heads for t0 and t1

Key Design Principles:
- All convolutions use CAUSAL padding (left padding only)
- Multi-head attention for feature fusion
- No bidirectional processing to prevent leakage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import FEATURE_CATEGORIES, get_feature_indices


class CausalConv1d(nn.Module):
    """1D Convolution with causal (left) padding to prevent future leakage."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,
            dilation=dilation
        )
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Causal (left) padding
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with:
    - Two causal convolutions with dilation
    - Residual connection
    - Layer normalization
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class TCNStream(nn.Module):
    """
    TCN stream for a single feature category.
    
    Input projection -> TCN blocks -> Output
    """
    
    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # TCN blocks with exponential dilation
        layers = []
        dilations = [2**i for i in range(num_layers)]
        
        for i, dilation in enumerate(dilations):
            layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))
        
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)
        return x


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention for temporal feature fusion.
    Uses causal mask to prevent attending to future timesteps.
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        seq_len = x.shape[1]
        causal_mask = self.generate_causal_mask(seq_len, x.device)
        
        # Self-attention with causal mask
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask)
        
        # Residual + norm
        x = self.norm(x + self.dropout(attn_out))
        return x


class CrossStreamAttention(nn.Module):
    """
    Attention mechanism to fuse multiple TCN streams with causal masking.
    
    IMPORTANT: With batch_first=True, MultiheadAttention operates over 
    the sequence dimension, so we MUST use a causal mask to prevent
    attending to future timesteps.
    """
    
    def __init__(self, stream_dim, num_streams, fusion_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Project each stream to same dimension
        self.stream_projections = nn.ModuleList([
            nn.Linear(stream_dim, fusion_dim // num_streams)
            for _ in range(num_streams)
        ])
        
        # Self-attention over concatenated streams (CAUSAL)
        self.attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask to prevent attending to future."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, streams):
        """
        Args:
            streams: list of (batch, seq_len, stream_dim) tensors
        Returns:
            fused: (batch, seq_len, fusion_dim)
        """
        # Project each stream
        projected = [proj(s) for proj, s in zip(self.stream_projections, streams)]
        
        # Concatenate along feature dimension
        fused = torch.cat(projected, dim=-1)  # (batch, seq_len, fusion_dim)
        
        # Generate causal mask to prevent future leakage
        seq_len = fused.shape[1]
        causal_mask = self.generate_causal_mask(seq_len, fused.device)
        
        # Self-attention with CAUSAL mask
        attn_out, _ = self.attention(fused, fused, fused, attn_mask=causal_mask)
        
        # Residual + norm
        fused = self.norm(fused + self.dropout(attn_out))
        
        return fused


class MultiStreamTCNGRUModel(nn.Module):
    """
    Multi-Stream TCN + GRU with Attention.
    
    Architecture:
    1. Split input into feature categories (prices, volumes, deltas, depth)
    2. Process each category through dedicated TCN stream
    3. Fuse streams using attention
    4. Temporal attention for sequence modeling
    5. GRU for final temporal processing
    6. Separate heads for t0 and t1 predictions
    """
    
    def __init__(
        self,
        input_dim=47,
        price_dim=16,
        volume_dim=16,
        delta_dim=11,
        depth_dim=4,
        tcn_hidden=32,
        tcn_layers=4,
        kernel_size=3,
        fusion_dim=128,
        num_attention_heads=4,
        attention_dropout=0.1,
        gru_hidden=128,
        gru_layers=2,
        output_dim=2,
        dropout=0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Feature indices for each category
        self.price_indices = get_feature_indices("prices")
        self.volume_indices = get_feature_indices("volumes")
        self.delta_indices = get_feature_indices("deltas")
        self.depth_indices = get_feature_indices("depth")
        
        # TCN streams for each feature category
        self.price_stream = TCNStream(price_dim, tcn_hidden, kernel_size, tcn_layers, dropout)
        self.volume_stream = TCNStream(volume_dim, tcn_hidden, kernel_size, tcn_layers, dropout)
        self.delta_stream = TCNStream(delta_dim, tcn_hidden, kernel_size, tcn_layers, dropout)
        self.depth_stream = TCNStream(depth_dim, tcn_hidden, kernel_size, tcn_layers, dropout)
        
        # Cross-stream attention fusion
        num_streams = 4
        self.stream_fusion = CrossStreamAttention(
            stream_dim=tcn_hidden,
            num_streams=num_streams,
            fusion_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            fusion_dim, num_attention_heads, attention_dropout
        )
        
        # GRU for final temporal modeling
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # MUST be unidirectional
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Separate output heads for t0 and t1
        self.head_t0 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1)
        )
        
        self.head_t1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, 47) input features
            hidden: Optional GRU hidden state
        
        Returns:
            output: (batch, seq_len, 2) predictions for t0, t1
            hidden: Updated GRU hidden state
        """
        # Split into feature categories
        price_features = x[:, :, self.price_indices]
        volume_features = x[:, :, self.volume_indices]
        delta_features = x[:, :, self.delta_indices]
        depth_features = x[:, :, self.depth_indices]
        
        # Process through TCN streams
        price_out = self.price_stream(price_features)
        volume_out = self.volume_stream(volume_features)
        delta_out = self.delta_stream(delta_features)
        depth_out = self.depth_stream(depth_features)
        
        # Fuse streams with attention
        streams = [price_out, volume_out, delta_out, depth_out]
        fused = self.stream_fusion(streams)
        
        # Temporal attention
        fused = self.temporal_attention(fused)
        
        # GRU processing
        gru_out, hidden = self.gru(fused, hidden)
        
        # Separate predictions for t0 and t1
        t0_pred = self.head_t0(gru_out)  # (batch, seq_len, 1)
        t1_pred = self.head_t1(gru_out)  # (batch, seq_len, 1)
        
        # Concatenate predictions
        output = torch.cat([t0_pred, t1_pred], dim=-1)  # (batch, seq_len, 2)
        
        return output, hidden


if __name__ == "__main__":
    # Test the model
    from config import get_model_kwargs
    
    batch_size = 4
    seq_len = 200
    
    model = MultiStreamTCNGRUModel(**get_model_kwargs(for_inference=False))
    x = torch.randn(batch_size, seq_len, 47)
    
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test feature indices
    print(f"\nFeature indices:")
    print(f"  Prices: {model.price_indices} ({len(model.price_indices)} features)")
    print(f"  Volumes: {model.volume_indices} ({len(model.volume_indices)} features)")
    print(f"  Deltas: {model.delta_indices} ({len(model.delta_indices)} features)")
    print(f"  Depth: {model.depth_indices} ({len(model.depth_indices)} features)")
