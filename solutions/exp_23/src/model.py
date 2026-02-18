"""
exp_23: TCN + GRU Model with Enhanced LOB Features

Architecture:
- Temporal Convolutional Network (TCN) with dilated causal convolutions
- GRU for final temporal modeling  
- Dual output heads for t0 and t1

Key Design Principles:
- All convolutions use CAUSAL padding (left padding only)
- Dilated convolutions for exponentially growing receptive field
- Residual connections for stable gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with causal (left) padding to prevent future leakage."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Causal padding size
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  # We'll apply padding manually
            dilation=dilation
        )
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Apply left (causal) padding only
        x = F.pad(x, (self.padding, 0))  # Pad left side only
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
        
        # Residual connection (1x1 conv if dimensions change)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch, seq_len, channels) for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # Back to (batch, channels, seq_len)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + residual


class TCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially increasing dilation.
    
    Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    With dilations [1, 2, 4, 8] and kernel_size=3:
    RF = 1 + 2 * 2 * 15 = 61 timesteps
    """
    
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        
        layers = []
        dilations = [2**i for i in range(num_layers)]  # [1, 2, 4, 8]
        
        for i, dilation in enumerate(dilations):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len) for Conv1d
        x = self.network(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        return x


class TCNGRUModel(nn.Module):
    """
    Combined TCN + GRU model for LOB prediction.
    
    Architecture:
    Input (batch, seq, features)
        → TCN (extract local patterns with large receptive field)
        → GRU (capture long-range temporal dependencies)
        → Output heads for t0, t1
    """
    
    def __init__(
        self,
        input_dim=47,  # 32 raw + 15 enhanced features
        tcn_hidden=64,
        tcn_layers=4,
        gru_hidden=128,
        gru_layers=2,
        output_dim=2,
        dropout=0.2
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, tcn_hidden)
        
        # TCN for local pattern extraction
        self.tcn = TCN(
            input_dim=tcn_hidden,
            hidden_dim=tcn_hidden,
            kernel_size=3,
            num_layers=tcn_layers,
            dropout=dropout
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=tcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # MUST be unidirectional to avoid leakage
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
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
            x: (batch, seq_len, input_dim) input features
            hidden: Optional GRU hidden state for stateful inference
        
        Returns:
            output: (batch, seq_len, 2) predictions for t0, t1
            hidden: Updated GRU hidden state
        """
        # Input projection
        x = self.input_proj(x)
        
        # TCN processing
        x = self.tcn(x)
        
        # GRU processing
        x, hidden = self.gru(x, hidden)
        
        # Output head
        output = self.output_head(x)
        
        return output, hidden


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 100
    input_dim = 47  # 32 raw + 15 enhanced
    
    model = TCNGRUModel(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
