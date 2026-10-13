"""
Solution for exp_10b: DeepLOB-style CNN-GRU with CAUSAL Convolutions

Inference wrapper for submission.
Uses raw 32 features (like exp_01) with CAUSAL CNN-GRU architecture.
exp_10b FIX: Uses causal convolutions (left-only padding) to prevent future leakage.
"""

from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model Architecture (included for standalone submission)
# ============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution - only looks at past and present, never future.
    
    Uses left-only padding: pad (kernel_size - 1) on the left, 0 on the right.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,
            dilation=dilation
        )
    
    def forward(self, x):
        x_padded = F.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        return out


class CausalInceptionModule(nn.Module):
    """
    CAUSAL Inception-style module with multiple kernel sizes.
    Uses left-only padding to prevent future leakage.
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
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.dropout(out)
        return out


class DeepLOBGRU(nn.Module):
    """DeepLOB-style CAUSAL CNN-GRU with dual heads."""
    
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
        
        # Total channels from inception = filters_per_branch * num_kernels
        inception1_out = inception1_filters * len(kernel_sizes)
        inception2_out = inception2_filters * len(kernel_sizes)
        
        # CAUSAL CNN Feature Extractor
        self.inception1 = CausalInceptionModule(input_dim, inception1_filters, kernel_sizes, dropout_cnn)
        self.inception2 = CausalInceptionModule(inception1_out, inception2_filters, kernel_sizes, dropout_cnn)
        
        # GRU
        self.gru = nn.GRU(
            input_size=inception2_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_gru if gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dual Heads
        self.head_t0 = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_t0),
            nn.Linear(head_hidden, 1)
        )
        
        self.head_t1 = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_t1),
            nn.Linear(head_hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, features) or (seq_len, features)
        Returns:
            pred_t0, pred_t1, hidden
        """
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, seq_len, _ = x.shape
        
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)
        
        # CNN feature extraction
        cnn_out = self.inception1(x_cnn)
        cnn_out = self.inception2(cnn_out)  # (batch, filters, seq_len)
        
        # Back to (batch, seq_len, filters)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # GRU
        gru_out, hidden = self.gru(cnn_out, hidden)
        
        # Dual heads
        pred_t0 = self.head_t0(gru_out)  # (batch, seq_len, 1)
        pred_t1 = self.head_t1(gru_out)
        
        return pred_t0, pred_t1, hidden


# ============================================================================
# Feature Definitions
# ============================================================================

# Raw features (32 total) - same as exp_01
PRICE_COLS = [f'p{i}' for i in range(12)]  # 12 price levels
VOLUME_COLS = [f'v{i}' for i in range(12)]  # 12 volume levels
DELTA_P_COLS = [f'dp{i}' for i in range(4)]  # 4 price deltas
DELTA_V_COLS = [f'dv{i}' for i in range(4)]  # 4 volume deltas

RAW_FEATURES = PRICE_COLS + VOLUME_COLS + DELTA_P_COLS + DELTA_V_COLS  # 32 features


# ============================================================================
# Solution Class
# ============================================================================

class Solution:
    """
    Inference solution for exp_10: DeepLOB-style CNN-GRU.
    
    Stateful inference - maintains GRU hidden state across calls.
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.hidden = None
        
        # Normalization parameters (will be loaded from checkpoint)
        self.means = None
        self.stds = None
        
        # Model config (defaults matching config.py, may be overridden by checkpoint)
        self.model_config = {
            'input_dim': 32,
            'inception1_filters': 16,
            'inception2_filters': 16,
            'kernel_sizes': [1, 3, 5],
            'gru_hidden': 128,
            'gru_layers': 2,
            'head_hidden': 64,
            'dropout_cnn': 0.0,   # No dropout at inference
            'dropout_gru': 0.0,
            'dropout_t0': 0.0,
            'dropout_t1': 0.0,
        }
    
    def init(self, **kwargs) -> None:
        """Initialize model and load weights."""
        
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Load checkpoint
        model_path = Path(__file__).parent / 'best_model.pth'
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config from checkpoint
        if 'model_config' in checkpoint:
            saved_config = checkpoint['model_config']
            # Update model config (but keep dropout at 0 for inference)
            for key in ['input_dim', 'inception1_filters', 'inception2_filters', 
                        'kernel_sizes', 'gru_hidden', 'gru_layers', 'head_hidden']:
                if key in saved_config:
                    self.model_config[key] = saved_config[key]
        
        # Create model
        self.model = DeepLOBGRU(**self.model_config)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load normalization parameters
        if 'normalization' in checkpoint:
            norm = checkpoint['normalization']
            self.means = torch.tensor(norm['means'], dtype=torch.float32, device=self.device)
            self.stds = torch.tensor(norm['stds'], dtype=torch.float32, device=self.device)
        else:
            # Default: no normalization
            self.means = torch.zeros(32, device=self.device)
            self.stds = torch.ones(32, device=self.device)
        
        # Initialize hidden state
        self.hidden = None
    
    def __call__(self, input_dict: dict) -> Tuple[float, float]:
        """
        Make prediction for a single timestep.
        
        Args:
            input_dict: Dictionary with keys p0-p11, v0-v11, dp0-dp3, dv0-dv3
            
        Returns:
            Tuple of (t0_prediction, t1_prediction)
        """
        
        # Extract features in correct order
        features = []
        for col in RAW_FEATURES:
            value = input_dict.get(col, 0.0)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            features.append(float(value))
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        # Normalize
        x = (x - self.means) / (self.stds + 1e-8)
        
        # Add batch and sequence dimensions: (1, 1, features)
        x = x.unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
        
        # Extract predictions
        t0 = pred_t0[0, 0, 0].item()
        t1 = pred_t1[0, 0, 0].item()
        
        return (t0, t1)
    
    def reset(self) -> None:
        """Reset hidden state for new sequence."""
        self.hidden = None


# ============================================================================
# Batch inference (for validation)
# ============================================================================

def batch_predict(model, data_df, device, means, stds):
    """
    Batch prediction for validation.
    
    Args:
        model: The DeepLOBGRU model
        data_df: DataFrame with raw features
        device: torch device
        means, stds: normalization parameters
        
    Returns:
        Array of predictions (N, 2) for t0 and t1
    """
    model.eval()
    
    # Extract features
    X = data_df[RAW_FEATURES].values.astype(np.float32)
    
    # Normalize
    X = (X - means) / (stds + 1e-8)
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Process in sequence (stateful)
    predictions = []
    hidden = None
    
    batch_size = 1000  # Process in chunks
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].unsqueeze(0)  # (1, chunk, features)
            pred_t0, pred_t1, hidden = model(batch, hidden)
            
            preds = torch.cat([pred_t0, pred_t1], dim=-1)  # (1, chunk, 2)
            predictions.append(preds.squeeze(0).cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Testing Solution class...")
    
    # Create solution
    solution = Solution()
    solution.init()
    
    print(f"Model loaded on device: {solution.device}")
    
    # Test with dummy input
    dummy_input = {col: 0.0 for col in RAW_FEATURES}
    dummy_input['p0'] = 100.0
    dummy_input['v0'] = 1000.0
    
    t0, t1 = solution(dummy_input)
    print(f"Test prediction: t0={t0:.4f}, t1={t1:.4f}")
    
    print("Solution test passed!")
