"""
exp_22: 2D Reshaped Temporal Convolution
Solution module for step-by-step inference

Note: This approach has a fundamental limitation for step-by-step inference:
The 2D structure requires the full 100-timestep window. For step-by-step,
we maintain a sliding buffer and re-process the entire window each step.
This is slower than GRU-based models but might have better accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


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
    """2D Reshaped Temporal Convolution Model."""
    
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
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_features, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        
        self.blocks_10x10 = nn.Sequential(
            *[ResBlock2D(hidden_channels, dropout) for _ in range(num_blocks)]
        )
        
        self.down_5x5 = DownsampleBlock(hidden_channels, hidden_channels * 2, dropout)
        
        self.blocks_5x5 = nn.Sequential(
            *[ResBlock2D(hidden_channels * 2, dropout) for _ in range(num_blocks)]
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.up_to_10x10 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        self.cell_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        
        self.pred_head = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
        
    def forward(self, x, hidden=None):
        batch_size, seq_len, num_features = x.shape
        
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, num_features, 10, 10)
        
        x = self.input_proj(x)
        
        feat_10x10 = self.blocks_10x10(x)
        
        x = self.down_5x5(feat_10x10)
        feat_5x5 = self.blocks_5x5(x)
        
        x = self.up_to_10x10(feat_5x5)
        x = x + feat_10x10
        
        x = self.cell_conv(x)
        
        pred = self.pred_head(x)
        
        pred = pred.view(batch_size, 2, 100)
        pred = pred.permute(0, 2, 1)
        
        pred_t0 = pred[:, :, 0]
        pred_t1 = pred[:, :, 1]
        
        return pred_t0, pred_t1, None


class PredictionModel:
    """
    Step-by-step prediction model for evaluation.
    
    Uses a sliding window buffer to enable step-by-step inference
    with the 2D temporal model.
    """
    
    def __init__(self):
        # Find model file
        solution_dir = Path(__file__).parent
        model_path = solution_dir / 'best_model.pth'
        
        # Set device
        self.device = torch.device('cpu')  # Use CPU for evaluation
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = Temporal2DModel(
            input_features=32,
            hidden_channels=config.get('hidden_channels', 64),
            num_blocks=config.get('num_blocks', 3),
            head_hidden=config.get('head_hidden', 128),
            dropout=0.0,  # No dropout at inference
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # State
        self.seq_len = 100
        self.buffer = np.zeros((self.seq_len, 32), dtype=np.float32)
        self.buffer_pos = 0
        self.current_stock_id = None
        self.current_day = None
        
        print(f"Loaded exp_22 model from {model_path}")
        print(f"  Valid corr avg: {checkpoint.get('valid_corr_avg', 'N/A'):.4f}")
    
    def reset_buffer(self):
        """Reset buffer for new sequence."""
        self.buffer = np.zeros((self.seq_len, 32), dtype=np.float32)
        self.buffer_pos = 0
    
    @torch.no_grad()
    def predict(self, features: np.ndarray, stock_id: int, day: int) -> tuple:
        """
        Predict for a single timestep.
        
        Args:
            features: (32,) array of features
            stock_id: stock identifier
            day: day identifier
            
        Returns:
            (pred_t0, pred_t1) tuple of predictions
        """
        # Reset buffer if stock/day changed
        if stock_id != self.current_stock_id or day != self.current_day:
            self.reset_buffer()
            self.current_stock_id = stock_id
            self.current_day = day
        
        # Add to buffer
        if self.buffer_pos < self.seq_len:
            self.buffer[self.buffer_pos] = features
            self.buffer_pos += 1
        else:
            # Shift buffer and add new observation
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = features
        
        # Prepare input: (1, 100, 32)
        x = torch.tensor(self.buffer, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0)
        
        # Forward pass
        pred_t0, pred_t1, _ = self.model(x)
        
        # Return prediction for current position
        pos = min(self.buffer_pos, self.seq_len) - 1
        return pred_t0[0, pos].item(), pred_t1[0, pos].item()
