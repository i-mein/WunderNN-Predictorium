"""
exp_22: 2D Reshaped Temporal Convolution
=========================================

Novel approach: Transform 1D temporal sequence to 2D for Conv2D processing.
- Input: (batch, 100, 32) -> (batch, 32, 10, 10)
- Treat time as 2D spatial structure
- Apply 2D convolutions to learn temporal patterns

Intuition:
- Time steps 0-9 form row 0, steps 10-19 form row 1, etc.
- This creates spatial locality in both x (within decade) and y (across decades)
- Conv2D can capture patterns like "what happens 10 steps after pattern X"

Architecture:
1. Reshape (100, 32) -> (32, 10, 10) (channels, height, width)
2. Initial projection + BatchNorm
3. ResNet-style 2D conv blocks
4. Global pooling + heads

Key innovation:
- The 2D structure preserves temporal ordering while enabling 2D receptive fields
- Adjacent cells in same row = consecutive timesteps (short-term)
- Adjacent cells in same column = timesteps 10 apart (medium-term)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """ResNet-style 2D convolutional block."""
    
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = F.gelu(out)
        
        return out


class DownsampleBlock(nn.Module):
    """Downsampling block with strided convolution."""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.skip_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        skip = self.skip(x)
        skip = self.skip_bn(skip)
        
        return out + skip


class Temporal2DModel(nn.Module):
    """
    2D Reshaped Temporal Convolution Model.
    
    Transforms temporal sequence (T=100, F=32) into 2D spatial (10x10x32)
    and applies 2D convolutions.
    """
    
    def __init__(
        self,
        input_features=32,
        hidden_channels=64,
        num_blocks=3,
        head_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        
        self.input_features = input_features
        
        # Initial projection from 32 channels to hidden_channels
        # Input: (batch, 32, 10, 10)
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_features, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        
        # ResNet-style blocks at 10x10 resolution
        self.blocks_10x10 = nn.Sequential(
            *[ResBlock2D(hidden_channels, dropout) for _ in range(num_blocks)]
        )
        
        # Downsample to 5x5
        self.down_5x5 = DownsampleBlock(hidden_channels, hidden_channels * 2, dropout)
        
        # ResNet-style blocks at 5x5 resolution
        self.blocks_5x5 = nn.Sequential(
            *[ResBlock2D(hidden_channels * 2, dropout) for _ in range(num_blocks)]
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # For per-timestep predictions, we need to maintain temporal structure
        # Use transposed convolutions to upsample back to 10x10
        self.up_to_10x10 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        # Final conv to produce per-cell predictions
        self.cell_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        
        # Per-position heads (each cell in 10x10 maps to a timestep)
        # Output is (batch, 2, 10, 10) -> reshape to (batch, 100, 2)
        self.pred_head = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)  # 2 = t0, t1
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq=100, features=32)
        
        Returns: pred_t0 (batch, 100), pred_t1 (batch, 100), None
        """
        batch_size, seq_len, num_features = x.shape
        
        # Reshape to 2D: (batch, 100, 32) -> (batch, 32, 10, 10)
        # Permute to (batch, features, time) then reshape time
        x = x.permute(0, 2, 1)  # (batch, 32, 100)
        x = x.view(batch_size, num_features, 10, 10)  # (batch, 32, 10, 10)
        
        # Initial projection
        x = self.input_proj(x)  # (batch, hidden, 10, 10)
        
        # Process at 10x10
        feat_10x10 = self.blocks_10x10(x)  # (batch, hidden, 10, 10)
        
        # Downsample and process at 5x5
        x = self.down_5x5(feat_10x10)  # (batch, hidden*2, 5, 5)
        feat_5x5 = self.blocks_5x5(x)  # (batch, hidden*2, 5, 5)
        
        # Upsample back to 10x10 for per-timestep predictions
        x = self.up_to_10x10(feat_5x5)  # (batch, hidden, 10, 10)
        
        # Skip connection from 10x10 features
        x = x + feat_10x10  # (batch, hidden, 10, 10)
        
        # Final convolution
        x = self.cell_conv(x)  # (batch, hidden//2, 10, 10)
        
        # Predictions
        pred = self.pred_head(x)  # (batch, 2, 10, 10)
        
        # Reshape back to temporal: (batch, 2, 10, 10) -> (batch, 100, 2)
        pred = pred.view(batch_size, 2, 100)  # (batch, 2, 100)
        pred = pred.permute(0, 2, 1)  # (batch, 100, 2)
        
        pred_t0 = pred[:, :, 0]  # (batch, 100)
        pred_t1 = pred[:, :, 1]  # (batch, 100)
        
        return pred_t0, pred_t1, None


class Temporal2DModelCausal(nn.Module):
    """
    Causal version of 2D Temporal Model for step-by-step inference.
    
    For online inference, we maintain a buffer of 100 timesteps.
    As new observations arrive, we shift the buffer and run the full model.
    
    Note: This is slower than GRU-based models for step-by-step inference
    because we need to process the entire 100-step window each time.
    
    Alternative: Use this model for batch inference only and combine
    with a simpler model for online predictions.
    """
    
    def __init__(
        self,
        input_features=32,
        hidden_channels=64,
        num_blocks=3,
        head_hidden=128,
        dropout=0.2,
        seq_len=100,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_features = input_features
        
        # Use the batch model
        self.model = Temporal2DModel(
            input_features=input_features,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            head_hidden=head_hidden,
            dropout=dropout,
        )
        
        # Buffer for step-by-step inference (filled with zeros initially)
        self.register_buffer('buffer', torch.zeros(1, seq_len, input_features))
        self.buffer_pos = 0
        
    def reset(self):
        """Reset buffer for new sequence."""
        self.buffer.zero_()
        self.buffer_pos = 0
    
    def forward_step(self, x):
        """
        Forward for a single timestep during online inference.
        
        x: (1, 1, 32) single observation
        
        Returns: pred_t0, pred_t1 for current timestep
        """
        # Add observation to buffer
        if self.buffer_pos < self.seq_len:
            self.buffer[0, self.buffer_pos] = x[0, 0]
            self.buffer_pos += 1
        else:
            # Shift buffer and add new observation
            self.buffer[0, :-1] = self.buffer[0, 1:].clone()
            self.buffer[0, -1] = x[0, 0]
        
        # Run full model on buffer
        pred_t0, pred_t1, _ = self.model(self.buffer)
        
        # Return prediction for most recent position
        pos = min(self.buffer_pos, self.seq_len) - 1
        return pred_t0[0, pos].item(), pred_t1[0, pos].item()
    
    def forward(self, x, hidden=None):
        """Batch forward pass."""
        return self.model(x, hidden)


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("2D Reshaped Temporal Model (exp_22)")
    print("=" * 60)
    
    # Test batch model
    model = Temporal2DModel()
    
    print("\nBatch model:")
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, _ = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  pred_t0 shape: {pred_t0.shape}")
    print(f"  pred_t1 shape: {pred_t1.shape}")
    
    # Test causal model
    print("\nCausal model (step-by-step):")
    causal_model = Temporal2DModelCausal()
    causal_model.eval()
    
    # Simulate step-by-step inference
    causal_model.reset()
    for i in range(5):
        x_step = torch.randn(1, 1, 32)
        t0, t1 = causal_model.forward_step(x_step)
        print(f"  Step {i}: pred_t0={t0:.4f}, pred_t1={t1:.4f}")
    
    print("\n" + "=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
