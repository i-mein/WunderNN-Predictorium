"""
DeepLOB-style Model Architecture for exp_10b

Based on:
- DeepLOB (Zhang et al., 2019) - "Deep Convolutional Neural Networks for Limit Order Books"
- arXiv:1808.03668

Key adaptations for our task:
1. CAUSAL Inception-style CNN modules (left-only padding, NO future leakage)
2. GRU (instead of LSTM) for faster training/inference
3. Dual heads for t0/t1 regression (instead of 3-class classification)
4. Heavy regularization to prevent overfitting (lesson from exp_09)
5. Unidirectional only (no future leakage)

exp_10b FIX: Uses causal convolutions with left-only padding to prevent future leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution - only looks at past and present, never future.
    
    Uses left-only padding: pad (kernel_size - 1) on the left, 0 on the right.
    Then trim the output to match input length.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Padding needed on left side only
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  # We'll pad manually
            dilation=dilation
        )
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Pad left only
        x_padded = F.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        return out


class CausalInceptionModule(nn.Module):
    """
    CAUSAL Inception-style module with multiple kernel sizes.
    
    Uses left-only padding to prevent future leakage:
    - kernel=1: Point-wise features (no padding needed)
    - kernel=3: Looks at t-2, t-1, t
    - kernel=5: Looks at t-4, t-3, t-2, t-1, t
    
    All branches are applied in parallel and concatenated.
    """
    
    def __init__(self, in_channels, out_channels_per_branch, kernel_sizes=[1, 3, 5], dropout=0.2):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                CausalConv1d(in_channels, out_channels_per_branch, kernel_size=k),
                nn.BatchNorm1d(out_channels_per_branch),
                nn.LeakyReLU(0.1),
            )
            self.branches.append(branch)
        
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels_per_branch * len(kernel_sizes)
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)  # Concatenate along channel dim
        out = self.dropout(out)
        return out


# Keep old InceptionModule for backward compatibility (but don't use it)
class InceptionModule(nn.Module):
    """
    NON-CAUSAL Inception module - DO NOT USE (has future leakage).
    Kept for backward compatibility only.
    """
    
    def __init__(self, in_channels, out_channels_per_branch, kernel_sizes=[1, 3, 5], dropout=0.2):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_channels_per_branch),
                nn.LeakyReLU(0.1),
            )
            self.branches.append(branch)
        
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels_per_branch * len(kernel_sizes)
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.dropout(out)
        return out


class DeepLOBGRU(nn.Module):
    """
    DeepLOB-style architecture: CAUSAL Inception CNN → GRU → Dual Heads
    
    Architecture:
    ```
    Input (batch, seq, 32 features)
            ↓
    [CAUSAL Inception 1]  ← 3 branches (k=1,3,5), 16 filters each = 48 channels
            ↓
    [CAUSAL Inception 2]  ← 3 branches (k=1,3,5), 16 filters each = 48 channels
            ↓
    [GRU]                 ← 2 layers, 128 hidden, unidirectional
            ↓
    ┌───────┴───────┐
    │               │
    [t0 Head]    [t1 Head]  ← Separate heads with different dropout
    │               │
    pred_t0      pred_t1
    ```
    
    Key design choices:
    - Raw 32 features (like exp_01, best LB)
    - CAUSAL convolutions (left-only padding, no future leakage)
    - Smaller model than exp_06 (GRU hidden 128 vs 256)
    - Heavy regularization (dropout, batch norm)
    - Unidirectional GRU (no future leakage)
    """
    
    def __init__(
        self,
        input_dim=32,
        inception1_filters=16,
        inception2_filters=16,
        kernel_sizes=[1, 3, 5],
        gru_hidden=128,
        gru_layers=2,
        head_hidden=64,
        dropout_cnn=0.2,
        dropout_gru=0.4,
        dropout_t0=0.3,
        dropout_t1=0.5,
    ):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # CAUSAL Inception Module 1
        self.inception1 = CausalInceptionModule(
            in_channels=input_dim,
            out_channels_per_branch=inception1_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout_cnn
        )
        inception1_out = inception1_filters * len(kernel_sizes)  # 48
        
        # CAUSAL Inception Module 2
        self.inception2 = CausalInceptionModule(
            in_channels=inception1_out,
            out_channels_per_branch=inception2_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout_cnn
        )
        inception2_out = inception2_filters * len(kernel_sizes)  # 48
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=inception2_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: No future leakage
            dropout=dropout_gru if gru_layers > 1 else 0.0
        )
        
        # Layer norm after GRU
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        # t0 head (easier target, less regularization)
        self.t0_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t0),
            nn.Linear(head_hidden, 1)
        )
        
        # t1 head (harder target, more regularization)
        self.t1_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t1),
            nn.Linear(head_hidden, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name.lower() or 'linear' in name.lower():
                    if param.dim() >= 2:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
                elif 'gru' in name.lower():
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: GRU hidden state from previous step (optional)
        
        Returns:
            pred_t0: Predictions for t0 (batch, seq_len, 1)
            pred_t1: Predictions for t1 (batch, seq_len, 1)
            hidden: Updated GRU hidden state
        """
        batch, seq_len, _ = x.shape
        
        # Normalize input
        x = self.input_norm(x)
        
        # Transpose for Conv1d: (batch, features, seq)
        x = x.transpose(1, 2)
        
        # Inception modules
        x = self.inception1(x)  # (batch, 48, seq)
        x = self.inception2(x)  # (batch, 48, seq)
        
        # Transpose back for GRU: (batch, seq, features)
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, hidden = self.gru(x, hidden)  # (batch, seq, gru_hidden)
        
        # Layer norm
        gru_out = self.gru_norm(gru_out)
        
        # Dual heads
        pred_t0 = self.t0_head(gru_out)  # (batch, seq, 1)
        pred_t1 = self.t1_head(gru_out)  # (batch, seq, 1)
        
        return pred_t0, pred_t1, hidden
    
    def forward_combined(self, x, hidden=None):
        """
        Forward pass returning combined output (for compatibility).
        
        Returns:
            predictions: (batch, seq_len, 2) with [t0, t1] concatenated
            hidden: Updated hidden state
        """
        pred_t0, pred_t1, hidden = self.forward(x, hidden)
        predictions = torch.cat([pred_t0, pred_t1], dim=-1)
        return predictions, hidden


class DeepLOBGRUWithAugmentation(DeepLOBGRU):
    """
    DeepLOB-GRU with input augmentation during training.
    
    Augmentation helps prevent overfitting:
    - Gaussian noise injection
    - Feature dropout
    """
    
    def __init__(
        self,
        input_dim=32,
        noise_std=0.01,
        feature_dropout=0.05,
        **kwargs
    ):
        super().__init__(input_dim=input_dim, **kwargs)
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
    
    def augment_input(self, x):
        """Apply augmentation during training."""
        if self.training:
            # Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
            
            # Feature dropout
            if self.feature_dropout > 0:
                mask = torch.rand_like(x) > self.feature_dropout
                x = x * mask
        
        return x
    
    def forward(self, x, hidden=None):
        x = self.augment_input(x)
        return super().forward(x, hidden)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with dummy data."""
    from config import MODEL_CONFIG
    
    model = DeepLOBGRU(
        input_dim=MODEL_CONFIG['input_dim'],
        inception1_filters=MODEL_CONFIG['inception1_filters'],
        inception2_filters=MODEL_CONFIG['inception2_filters'],
        kernel_sizes=MODEL_CONFIG['kernel_sizes'],
        gru_hidden=MODEL_CONFIG['gru_hidden'],
        gru_layers=MODEL_CONFIG['gru_layers'],
        head_hidden=MODEL_CONFIG['head_hidden'],
        dropout_cnn=MODEL_CONFIG['dropout_cnn'],
        dropout_gru=MODEL_CONFIG['dropout_gru'],
        dropout_t0=MODEL_CONFIG['dropout_t0'],
        dropout_t1=MODEL_CONFIG['dropout_t1'],
    )
    
    print(f"Model: DeepLOBGRU")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 100
    x = torch.randn(batch_size, seq_len, MODEL_CONFIG['input_dim'])
    
    model.eval()
    with torch.no_grad():
        pred_t0, pred_t1, hidden = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"pred_t0 shape: {pred_t0.shape}")
    print(f"pred_t1 shape: {pred_t1.shape}")
    print(f"hidden shape: {hidden.shape}")
    
    # Test with augmentation
    model_aug = DeepLOBGRUWithAugmentation(
        input_dim=MODEL_CONFIG['input_dim'],
        noise_std=0.01,
        feature_dropout=0.05,
        inception1_filters=MODEL_CONFIG['inception1_filters'],
        inception2_filters=MODEL_CONFIG['inception2_filters'],
        kernel_sizes=MODEL_CONFIG['kernel_sizes'],
        gru_hidden=MODEL_CONFIG['gru_hidden'],
        gru_layers=MODEL_CONFIG['gru_layers'],
        head_hidden=MODEL_CONFIG['head_hidden'],
        dropout_cnn=MODEL_CONFIG['dropout_cnn'],
        dropout_gru=MODEL_CONFIG['dropout_gru'],
        dropout_t0=MODEL_CONFIG['dropout_t0'],
        dropout_t1=MODEL_CONFIG['dropout_t1'],
    )
    
    print(f"\nModel with Augmentation: {count_parameters(model_aug):,} params")
    
    return model


if __name__ == "__main__":
    test_model()
