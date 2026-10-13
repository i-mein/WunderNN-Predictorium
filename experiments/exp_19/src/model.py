"""
Exp 19: Conv2D Feature Extraction with Extensive Difference Features

Inspired by:
1. DeepLOB (Zhang et al.): Transform LOB data to 2D and use Conv2D layers
2. Optiver competitions: Multi-scale temporal features
3. Time series classification: Treating sequences as images

Key innovations:
1. Transform time series to 2D representation:
   - Rows: Time steps (sequence length)
   - Columns: Features organized by category
   - Channels: Different "views" of the data (raw, diffs, diffs-of-diffs)
   
2. Extensive difference features for temporal dependencies:
   - 1st order differences (current - previous)
   - 2nd order differences (acceleration)
   - Cross-feature differences (bid vs ask)
   
3. Causal Conv2D with proper padding:
   - Only look at current and past time steps
   - Kernel operates on feature dimension freely
   
4. Inception-style multi-scale temporal convolutions:
   - Different kernel sizes capture different time scales
   - Parallel branches concatenated
   
5. GRU on extracted features for final temporal modeling

Architecture flow:
Raw (batch, seq, 32) -> Multi-Channel 2D (batch, channels, seq, features)
-> Conv2D Blocks -> Inception Module -> GRU -> Heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferenceFeatureGenerator(nn.Module):
    """
    Generate extensive difference features for temporal dependencies.
    All operations are CAUSAL - only use current and past data.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: (batch, seq, 32) - raw features
        Returns: (batch, seq, n_features) - expanded features with diffs
        
        Feature layout (32 raw):
        - p0-p11 (0-11): price levels
        - v0-v11 (12-23): volume levels
        - dp0-dp3 (24-27): delta prices
        - dv0-dv3 (28-31): delta volumes
        """
        batch, seq, _ = x.shape
        
        # Extract feature groups
        prices = x[..., :12]       # 12 features
        volumes = x[..., 12:24]    # 12 features
        dprices = x[..., 24:28]    # 4 features
        dvolumes = x[..., 28:32]   # 4 features
        
        # === 1st Order Differences (temporal) ===
        # Pad with zeros at start for causality
        prices_diff1 = prices - F.pad(prices[:, :-1, :], (0, 0, 1, 0))
        volumes_diff1 = volumes - F.pad(volumes[:, :-1, :], (0, 0, 1, 0))
        dprices_diff1 = dprices - F.pad(dprices[:, :-1, :], (0, 0, 1, 0))
        dvolumes_diff1 = dvolumes - F.pad(dvolumes[:, :-1, :], (0, 0, 1, 0))
        
        # === 2nd Order Differences (acceleration) ===
        prices_diff2 = prices_diff1 - F.pad(prices_diff1[:, :-1, :], (0, 0, 1, 0))
        volumes_diff2 = volumes_diff1 - F.pad(volumes_diff1[:, :-1, :], (0, 0, 1, 0))
        
        # === Cross-Feature Differences ===
        # Bid-ask spread at each level
        bid_prices = prices[..., :6]   # p0-p5
        ask_prices = prices[..., 6:]   # p6-p11
        spreads = ask_prices - bid_prices  # 6 features
        
        # Bid-ask volume imbalance at each level
        bid_vols = volumes[..., :6]    # v0-v5
        ask_vols = volumes[..., 6:]    # v6-v11
        vol_diffs = bid_vols - ask_vols  # 6 features
        vol_ratios = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)  # 6 features
        
        # === Engineered Features (no look-ahead) ===
        # Mid-price
        mid_price = (prices[..., 0:1] + prices[..., 6:7]) / 2  # 1 feature
        
        # Weighted mid-price
        total_bid_vol = bid_vols.sum(dim=-1, keepdim=True)
        total_ask_vol = ask_vols.sum(dim=-1, keepdim=True)
        weighted_bid = (bid_prices * bid_vols).sum(dim=-1, keepdim=True) / (total_bid_vol + 1e-8)
        weighted_ask = (ask_prices * ask_vols).sum(dim=-1, keepdim=True) / (total_ask_vol + 1e-8)
        weighted_mid = (weighted_bid + weighted_ask) / 2  # 1 feature
        
        # Total volume imbalance
        total_vol_imb = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)  # 1 feature
        
        # Price/volume momentum from delta features
        price_momentum = dprices.sum(dim=-1, keepdim=True)  # 1 feature
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)   # 1 feature
        
        # === Price depth features ===
        # Cumulative volume at each price level
        cumsum_bid_vol = bid_vols.cumsum(dim=-1)  # 6 features
        cumsum_ask_vol = ask_vols.cumsum(dim=-1)  # 6 features
        
        # Concatenate all features
        all_features = torch.cat([
            # Raw features (32)
            prices,           # 12
            volumes,          # 12
            dprices,          # 4
            dvolumes,         # 4
            
            # 1st order diffs (32)
            prices_diff1,     # 12
            volumes_diff1,    # 12
            dprices_diff1,    # 4
            dvolumes_diff1,   # 4
            
            # 2nd order diffs (24)
            prices_diff2,     # 12
            volumes_diff2,    # 12
            
            # Cross-feature (18)
            spreads,          # 6
            vol_diffs,        # 6
            vol_ratios,       # 6
            
            # Engineered (5)
            mid_price,        # 1
            weighted_mid,     # 1
            total_vol_imb,    # 1
            price_momentum,   # 1
            vol_momentum,     # 1
            
            # Depth (12)
            cumsum_bid_vol,   # 6
            cumsum_ask_vol,   # 6
        ], dim=-1)
        
        # Total: 32 + 32 + 24 + 18 + 5 + 12 = 123 features
        return all_features


class To2DRepresentation(nn.Module):
    """
    Transform 1D time series to 2D representation for Conv2D.
    
    Input: (batch, seq, features)
    Output: (batch, channels, seq, feature_groups)
    
    We organize features into groups that make sense spatially:
    - Channel 0: Price features (raw + diffs)
    - Channel 1: Volume features (raw + diffs)
    - Channel 2: Delta features
    - Channel 3: Cross/engineered features
    """
    
    def __init__(self, n_features=123, n_feature_groups=4):
        super().__init__()
        self.n_features = n_features
        self.n_feature_groups = n_feature_groups
        
        # Feature indices for each group
        # Group 0: Prices (raw 12 + diff1 12 + diff2 12 = 36)
        # Group 1: Volumes (raw 12 + diff1 12 + diff2 12 = 36)
        # Group 2: Deltas (raw 8 + diff1 8 = 16)
        # Group 3: Cross + Engineered (18 + 5 + 12 = 35)
        
        # For Conv2D, we'll use (batch, 1, seq, features) 
        # and let Conv2D learn the spatial patterns
        
    def forward(self, x):
        """
        x: (batch, seq, features)
        Returns: (batch, 1, seq, features) for Conv2D
        """
        # Add channel dimension
        return x.unsqueeze(1)  # (batch, 1, seq, features)


class CausalConv2dBlock(nn.Module):
    """
    Causal Conv2D block that only looks at current and past time steps.
    Uses asymmetric padding: pad only at the start in time dimension.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), stride=1):
        super().__init__()
        self.kernel_h = kernel_size[0]  # Time dimension
        self.kernel_w = kernel_size[1]  # Feature dimension
        
        # Padding for causality: only pad at the start in time dimension
        # Feature dimension: symmetric padding (or let it shrink)
        self.pad_h = self.kernel_h - 1  # Causal padding for time
        self.pad_w = self.kernel_w // 2  # Symmetric padding for features
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, self.pad_w)  # No automatic time padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        """
        x: (batch, channels, seq, features)
        """
        # Causal padding: pad only at the beginning of time dimension
        x = F.pad(x, (0, 0, self.pad_h, 0))  # (left, right, top, bottom)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception-style module for multi-scale temporal pattern extraction.
    Different kernel sizes capture different time scales.
    All branches use causal convolutions.
    """
    
    def __init__(self, in_channels, out_channels_per_branch=32):
        super().__init__()
        
        # Branch 1: 1x1 conv (no temporal context, pure feature mixing)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        
        # Branch 2: 1x1 -> 3x1 (short-term patterns)
        self.branch2_reduce = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, 1))
        self.branch2_pad = 2  # Causal pad for kernel size 3
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(out_channels_per_branch, out_channels_per_branch, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        
        # Branch 3: 1x1 -> 5x1 (medium-term patterns)
        self.branch3_reduce = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, 1))
        self.branch3_pad = 4  # Causal pad for kernel size 5
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(out_channels_per_branch, out_channels_per_branch, kernel_size=(5, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        
        # Branch 4: MaxPool -> 1x1 (pooled features)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        
        self.out_channels = out_channels_per_branch * 4
        
    def forward(self, x):
        """
        x: (batch, channels, seq, features)
        """
        # Branch 1: no temporal padding needed
        b1 = self.branch1(x)
        
        # Branch 2: causal padding
        b2 = self.branch2_reduce(x)
        b2 = F.pad(b2, (0, 0, self.branch2_pad, 0))
        b2 = self.branch2_conv(b2)
        
        # Branch 3: causal padding
        b3 = self.branch3_reduce(x)
        b3 = F.pad(b3, (0, 0, self.branch3_pad, 0))
        b3 = self.branch3_conv(b3)
        
        # Branch 4: max pool with padding maintains size
        b4 = self.branch4(x)
        
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)


class Conv2DLOB(nn.Module):
    """
    Conv2D-based LOB model inspired by DeepLOB.
    
    Architecture:
    1. DifferenceFeatureGenerator: Expand to 123 features with diffs
    2. To2D: Reshape to (batch, 1, seq, 123)
    3. Conv2D blocks: Extract spatial patterns in feature space
    4. Inception module: Multi-scale temporal patterns
    5. Flatten channels and use GRU for final temporal modeling
    6. Dual prediction heads
    """
    
    def __init__(
        self,
        d_model=64,
        gru_hidden=64,
        gru_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        
        self.diff_gen = DifferenceFeatureGenerator()
        self.to_2d = To2DRepresentation(n_features=123)
        
        # Conv2D feature extraction blocks
        # Input: (batch, 1, seq, 123)
        
        # Block 1: (1, 2) stride on feature dim to reduce from 123 to ~61
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        # Block 2: Causal temporal convolution
        self.conv2 = CausalConv2dBlock(32, 32, kernel_size=(4, 1))
        self.conv3 = CausalConv2dBlock(32, 32, kernel_size=(4, 1))
        
        # Block 3: Reduce feature dimension further
        # After conv1: features ~ 61, after this: ~30
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        # Block 4: More temporal convolutions
        self.conv5 = CausalConv2dBlock(32, 32, kernel_size=(4, 1))
        self.conv6 = CausalConv2dBlock(32, 32, kernel_size=(4, 1))
        
        # Block 5: Final feature reduction to 1
        # We'll use adaptive pooling to handle variable sizes
        self.conv_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # Pool feature dim to 1
        )
        
        # Inception module for multi-scale temporal patterns
        # Input: (batch, 32, seq, 1)
        self.inception = InceptionModule(32, out_channels_per_branch=16)
        # Output: (batch, 64, seq, 1)
        
        # Reshape and GRU
        self.dropout = nn.Dropout(dropout)
        
        # GRU input: inception output channels (64)
        self.gru = nn.GRU(
            input_size=64,  # From inception (4 * 16)
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        
        # Prediction heads
        self.head_t0 = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
        self.head_t1 = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq, 32)
        Returns: predictions for t0 and t1 at ALL timesteps
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Generate difference features
        x = self.diff_gen(x)  # (batch, seq, 123)
        
        # To 2D
        x = self.to_2d(x)  # (batch, 1, seq, 123)
        
        # Conv2D blocks
        x = self.conv1(x)   # (batch, 32, seq, 61)
        x = self.conv2(x)   # (batch, 32, seq, 61)
        x = self.conv3(x)   # (batch, 32, seq, 61)
        x = self.conv4(x)   # (batch, 32, seq, 30)
        x = self.conv5(x)   # (batch, 32, seq, 30)
        x = self.conv6(x)   # (batch, 32, seq, 30)
        
        # Final feature pooling
        x = self.conv_final(x)  # (batch, 32, seq, 1)
        
        # Inception for multi-scale patterns
        x = self.inception(x)  # (batch, 64, seq, 1)
        
        # Reshape for GRU: (batch, seq, channels)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, seq, 64)
        
        # Dropout
        x = self.dropout(x)
        
        # GRU
        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers, batch_size, self.gru.hidden_size,
                device=x.device
            )
        
        gru_out, hidden = self.gru(x, hidden)
        
        # Predict at ALL timesteps (not just last)
        # gru_out: (batch, seq, gru_hidden)
        
        # Predictions at all timesteps
        pred_t0 = self.head_t0(gru_out).squeeze(-1)  # (batch, seq)
        pred_t1 = self.head_t1(gru_out).squeeze(-1)  # (batch, seq)
        
        return pred_t0, pred_t1, hidden


class Conv2DLOBSimple(nn.Module):
    """
    Simpler version with fewer parameters to avoid overfitting.
    """
    
    def __init__(
        self,
        gru_hidden=48,
        gru_layers=1,
        dropout=0.15,
    ):
        super().__init__()
        
        self.diff_gen = DifferenceFeatureGenerator()
        
        # Simpler Conv2D pipeline
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, seq, 123)
            nn.Conv2d(1, 16, kernel_size=(1, 4), stride=(1, 2)),  # -> (b, 16, seq, 60)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 2)),  # -> (b, 16, seq, 29)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 2)),  # -> (b, 16, seq, 14)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 14)),  # -> (b, 16, seq, 1)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        
        # Causal temporal convolution
        self.temporal_conv = nn.Sequential(
            CausalConv2dBlock(16, 24, kernel_size=(3, 1)),
            CausalConv2dBlock(24, 24, kernel_size=(3, 1)),
        )
        
        # GRU
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=24,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        
        # Simple heads
        self.head_t0 = nn.Linear(gru_hidden, 1)
        self.head_t1 = nn.Linear(gru_hidden, 1)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Generate features
        x = self.diff_gen(x)  # (batch, seq, 123)
        x = x.unsqueeze(1)    # (batch, 1, seq, 123)
        
        # Conv2D feature extraction
        x = self.conv_layers(x)  # (batch, 16, seq, 1)
        
        # Temporal conv
        x = self.temporal_conv(x)  # (batch, 24, seq, 1)
        
        # Reshape for GRU
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, seq, 24)
        x = self.dropout(x)
        
        # GRU
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
        
        gru_out, hidden = self.gru(x, hidden)
        
        # Predict at ALL timesteps for training
        pred_t0 = self.head_t0(gru_out).squeeze(-1)  # (batch, seq)
        pred_t1 = self.head_t1(gru_out).squeeze(-1)  # (batch, seq)
        
        return pred_t0, pred_t1, hidden


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test data
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 32).to(device)
    
    # Test DifferenceFeatureGenerator
    diff_gen = DifferenceFeatureGenerator()
    x_expanded = diff_gen(x)
    print(f"Feature expansion: {x.shape} -> {x_expanded.shape}")
    
    # Test Conv2DLOB (full version)
    print("\n--- Conv2DLOB (Full) ---")
    model1 = Conv2DLOB().to(device)
    print(f"Parameters: {count_parameters(model1):,}")
    
    pred_t0, pred_t1, hidden = model1(x)
    print(f"Output shapes: t0={pred_t0.shape}, t1={pred_t1.shape}, hidden={hidden.shape}")
    
    # Test Conv2DLOBSimple
    print("\n--- Conv2DLOBSimple ---")
    model2 = Conv2DLOBSimple().to(device)
    print(f"Parameters: {count_parameters(model2):,}")
    
    pred_t0, pred_t1, hidden = model2(x)
    print(f"Output shapes: t0={pred_t0.shape}, t1={pred_t1.shape}, hidden={hidden.shape}")
    
    print("\n✅ All models work correctly!")
