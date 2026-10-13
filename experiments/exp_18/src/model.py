"""
Exp 18: Multi-Stream Hybrid Architecture

Key design principles:
1. Separate processing streams for each feature category
2. ALL features are leakage-free (only use current and past data)
3. Simpler model to avoid overfitting (learned from exp_17)
4. Stream outputs concatenated before prediction heads

Feature Streams (32 raw features):
- Stream 1: Price levels (p0-p11) - 12 features
- Stream 2: Volume levels (v0-v11) - 12 features  
- Stream 3: Delta prices (dp0-dp3) - 4 features
- Stream 4: Delta volumes (dv0-dv3) - 4 features

Engineered features (leakage-free, computed from current step only):
- Mid-price: (p0 + p6) / 2
- Spread: p6 - p0
- Volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
- Best level imbalance: (v0 - v6) / (v0 + v6)
- Price momentum: sum(dp0-dp3)
- Volume momentum: sum(dv0-dv3)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakageFreeFeatures(nn.Module):
    """
    Compute engineered features from CURRENT STEP ONLY.
    No rolling windows, no future data, no cross-sequence data.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: (batch, seq, 32)
        Returns: (batch, seq, 6) engineered features
        """
        # Extract feature groups
        prices = x[..., :12]      # p0-p11
        volumes = x[..., 12:24]   # v0-v11
        dprices = x[..., 24:28]   # dp0-dp3
        dvolumes = x[..., 28:32]  # dv0-dv3
        
        # Best bid/ask prices and volumes
        p0 = prices[..., 0:1]   # Best bid price
        p6 = prices[..., 6:7]   # Best ask price
        v0 = volumes[..., 0:1]  # Best bid volume
        v6 = volumes[..., 6:7]  # Best ask volume
        
        # 1. Mid-price (normalized by spread)
        mid_price = (p0 + p6) / 2
        
        # 2. Spread
        spread = p6 - p0
        
        # 3. Total volume imbalance (bid vs ask)
        bid_vol = volumes[..., :6].sum(dim=-1, keepdim=True)
        ask_vol = volumes[..., 6:].sum(dim=-1, keepdim=True)
        vol_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        
        # 4. Best level volume imbalance
        best_vol_imbalance = (v0 - v6) / (v0 + v6 + 1e-8)
        
        # 5. Price momentum (sum of recent price changes)
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        
        # 6. Volume momentum (sum of recent volume changes)
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        engineered = torch.cat([
            mid_price,           # 1
            spread,              # 1
            vol_imbalance,       # 1
            best_vol_imbalance,  # 1
            price_momentum,      # 1
            vol_momentum,        # 1
        ], dim=-1)
        
        return engineered  # (batch, seq, 6)


class StreamEncoder(nn.Module):
    """
    Encode a single feature stream using projection + GRU.
    Simple and effective - avoids transformer overhead.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.gru = nn.GRU(
            hidden_dim,
            output_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        self.output_dim = output_dim
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq, input_dim)
        Returns: output (batch, seq, output_dim), hidden
        """
        h = self.input_proj(x)
        output, hidden = self.gru(h, hidden)
        return output, hidden


class MultiStreamHybrid(nn.Module):
    """
    Multi-stream architecture with separate encoders for each feature category.
    
    Architecture:
    1. Price stream: p0-p11 (12 features) → GRU → 64 dim
    2. Volume stream: v0-v11 (12 features) → GRU → 64 dim
    3. Delta price stream: dp0-dp3 (4 features) → GRU → 32 dim
    4. Delta volume stream: dv0-dv3 (4 features) → GRU → 32 dim
    5. Engineered stream: 6 features → Linear → 32 dim
    
    Total: 64 + 64 + 32 + 32 + 32 = 224 dim → Prediction heads
    """
    
    def __init__(
        self,
        price_hidden=48,
        price_output=64,
        volume_hidden=48,
        volume_output=64,
        delta_hidden=32,
        delta_output=32,
        eng_output=32,
        head_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        
        # Feature engineering (leakage-free)
        self.feature_eng = LeakageFreeFeatures()
        
        # Stream 1: Price levels (12 features)
        self.price_stream = StreamEncoder(
            input_dim=12,
            hidden_dim=price_hidden,
            output_dim=price_output,
            num_layers=2,
            dropout=dropout,
        )
        
        # Stream 2: Volume levels (12 features)
        self.volume_stream = StreamEncoder(
            input_dim=12,
            hidden_dim=volume_hidden,
            output_dim=volume_output,
            num_layers=2,
            dropout=dropout,
        )
        
        # Stream 3: Delta prices (4 features)
        self.dprice_stream = StreamEncoder(
            input_dim=4,
            hidden_dim=delta_hidden,
            output_dim=delta_output,
            num_layers=1,
            dropout=dropout,
        )
        
        # Stream 4: Delta volumes (4 features)
        self.dvolume_stream = StreamEncoder(
            input_dim=4,
            hidden_dim=delta_hidden,
            output_dim=delta_output,
            num_layers=1,
            dropout=dropout,
        )
        
        # Stream 5: Engineered features (6 features)
        self.eng_proj = nn.Sequential(
            nn.Linear(6, eng_output),
            nn.LayerNorm(eng_output),
            nn.GELU(),
        )
        
        # Combined dimension
        combined_dim = price_output + volume_output + delta_output * 2 + eng_output
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # t0 head (prices are most important)
        self.t0_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        # t1 head (deltas are most important)
        self.t1_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        # Store dimensions for hidden state management
        self.price_output = price_output
        self.volume_output = volume_output
        self.delta_output = delta_output
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq, 32) raw features
        hidden: dict of hidden states for each stream
        
        Returns: pred_t0, pred_t1, hidden
        """
        batch_size = x.shape[0]
        
        # Split raw features into streams
        prices = x[..., :12]       # p0-p11
        volumes = x[..., 12:24]    # v0-v11
        dprices = x[..., 24:28]    # dp0-dp3
        dvolumes = x[..., 28:32]   # dv0-dv3
        
        # Compute engineered features
        eng_features = self.feature_eng(x)  # (batch, seq, 6)
        
        # Initialize hidden states if needed
        if hidden is None:
            hidden = {
                'price': None,
                'volume': None,
                'dprice': None,
                'dvolume': None,
            }
        
        # Process each stream
        price_out, hidden['price'] = self.price_stream(prices, hidden['price'])
        volume_out, hidden['volume'] = self.volume_stream(volumes, hidden['volume'])
        dprice_out, hidden['dprice'] = self.dprice_stream(dprices, hidden['dprice'])
        dvolume_out, hidden['dvolume'] = self.dvolume_stream(dvolumes, hidden['dvolume'])
        
        # Process engineered features (no hidden state needed)
        eng_out = self.eng_proj(eng_features)
        
        # Concatenate all stream outputs
        combined = torch.cat([
            price_out,    # 64
            volume_out,   # 64
            dprice_out,   # 32
            dvolume_out,  # 32
            eng_out,      # 32
        ], dim=-1)  # Total: 224
        
        # Fusion
        fused = self.fusion(combined)
        
        # Predictions
        pred_t0 = self.t0_head(fused).squeeze(-1)
        pred_t1 = self.t1_head(fused).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


if __name__ == "__main__":
    # Test model
    model = MultiStreamHybrid()
    
    print("=" * 60)
    print("Multi-Stream Hybrid Model (exp_18)")
    print("=" * 60)
    
    print("\nTraining mode (full sequence):")
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  pred_t0 shape: {pred_t0.shape}")
    print(f"  pred_t1 shape: {pred_t1.shape}")
    
    print("\nInference mode (single step):")
    x_step = torch.randn(1, 1, 32)
    pred_t0_step, pred_t1_step, hidden_step = model(x_step, hidden=None)
    print(f"  Input shape: {x_step.shape}")
    print(f"  pred_t0 shape: {pred_t0_step.shape}")
    print(f"  pred_t1 shape: {pred_t1_step.shape}")
    
    print("\nInference mode (with carried hidden state):")
    pred_t0_step2, pred_t1_step2, hidden_step2 = model(x_step, hidden=hidden_step)
    print(f"  Hidden state carried successfully")
    
    print("\n" + "=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
