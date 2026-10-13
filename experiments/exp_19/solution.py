"""
Exp 19: Conv2D LOB Solution

Solution file for submission - standalone single-step inference.

Key features:
1. Conv2D feature extraction from 2D representation of LOB
2. Extensive difference features (1st, 2nd order, cross-feature)
3. Causal convolutions for temporal patterns
4. GRU for final sequence modeling
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass  
class DataPoint:
    """Data point for inference."""
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


# ============================================================================
# Model Definition (copied from src/model.py for standalone submission)
# ============================================================================

class DifferenceFeatureGenerator(nn.Module):
    """
    Generate extensive difference features for temporal dependencies.
    For single-step inference, we need to track previous values.
    """
    
    def __init__(self):
        super().__init__()
        # For single-step inference, we'll store previous values
        self.prev_features = None
        self.prev_diff1 = None
    
    def reset(self):
        """Reset state for new sequence."""
        self.prev_features = None
        self.prev_diff1 = None
    
    def forward(self, x):
        """
        x: (batch, seq, 32) or (batch, 1, 32) for single-step
        Returns: (batch, seq, 123) expanded features
        """
        batch, seq, _ = x.shape
        
        # Extract feature groups
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        # For single-step inference with seq=1
        if seq == 1 and self.prev_features is not None:
            # Compute diffs from previous
            prev_prices = self.prev_features[..., :12]
            prev_volumes = self.prev_features[..., 12:24]
            prev_dprices = self.prev_features[..., 24:28]
            prev_dvolumes = self.prev_features[..., 28:32]
            
            prices_diff1 = prices - prev_prices
            volumes_diff1 = volumes - prev_volumes
            dprices_diff1 = dprices - prev_dprices
            dvolumes_diff1 = dvolumes - prev_dvolumes
            
            # 2nd order diffs
            if self.prev_diff1 is not None:
                prices_diff2 = prices_diff1 - self.prev_diff1['prices']
                volumes_diff2 = volumes_diff1 - self.prev_diff1['volumes']
            else:
                prices_diff2 = torch.zeros_like(prices_diff1)
                volumes_diff2 = torch.zeros_like(volumes_diff1)
            
            # Store for next step
            self.prev_diff1 = {
                'prices': prices_diff1.clone(),
                'volumes': volumes_diff1.clone(),
            }
        else:
            # First step or batch mode: use zero diffs
            if seq == 1:
                # First step: all diffs are zero
                prices_diff1 = torch.zeros_like(prices)
                volumes_diff1 = torch.zeros_like(volumes)
                dprices_diff1 = torch.zeros_like(dprices)
                dvolumes_diff1 = torch.zeros_like(dvolumes)
                prices_diff2 = torch.zeros_like(prices)
                volumes_diff2 = torch.zeros_like(volumes)
            else:
                # Batch mode with full sequence
                prices_diff1 = prices - F.pad(prices[:, :-1, :], (0, 0, 1, 0))
                volumes_diff1 = volumes - F.pad(volumes[:, :-1, :], (0, 0, 1, 0))
                dprices_diff1 = dprices - F.pad(dprices[:, :-1, :], (0, 0, 1, 0))
                dvolumes_diff1 = dvolumes - F.pad(dvolumes[:, :-1, :], (0, 0, 1, 0))
                
                prices_diff2 = prices_diff1 - F.pad(prices_diff1[:, :-1, :], (0, 0, 1, 0))
                volumes_diff2 = volumes_diff1 - F.pad(volumes_diff1[:, :-1, :], (0, 0, 1, 0))
        
        # Store current for next step (single-step mode)
        if seq == 1:
            self.prev_features = x.clone()
        
        # Cross-feature differences
        bid_prices = prices[..., :6]
        ask_prices = prices[..., 6:]
        spreads = ask_prices - bid_prices
        
        bid_vols = volumes[..., :6]
        ask_vols = volumes[..., 6:]
        vol_diffs = bid_vols - ask_vols
        vol_ratios = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)
        
        # Engineered features
        mid_price = (prices[..., 0:1] + prices[..., 6:7]) / 2
        
        total_bid_vol = bid_vols.sum(dim=-1, keepdim=True)
        total_ask_vol = ask_vols.sum(dim=-1, keepdim=True)
        weighted_bid = (bid_prices * bid_vols).sum(dim=-1, keepdim=True) / (total_bid_vol + 1e-8)
        weighted_ask = (ask_prices * ask_vols).sum(dim=-1, keepdim=True) / (total_ask_vol + 1e-8)
        weighted_mid = (weighted_bid + weighted_ask) / 2
        
        total_vol_imb = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)
        
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        cumsum_bid_vol = bid_vols.cumsum(dim=-1)
        cumsum_ask_vol = ask_vols.cumsum(dim=-1)
        
        # Concatenate all: 32 + 32 + 24 + 18 + 5 + 12 = 123
        all_features = torch.cat([
            prices, volumes, dprices, dvolumes,
            prices_diff1, volumes_diff1, dprices_diff1, dvolumes_diff1,
            prices_diff2, volumes_diff2,
            spreads, vol_diffs, vol_ratios,
            mid_price, weighted_mid, total_vol_imb, price_momentum, vol_momentum,
            cumsum_bid_vol, cumsum_ask_vol,
        ], dim=-1)
        
        return all_features


class CausalConv2dBlock(nn.Module):
    """Causal Conv2D block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=(4, 1)):
        super().__init__()
        self.kernel_h = kernel_size[0]
        self.pad_h = self.kernel_h - 1
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_h, 0))
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv2DLOBSimple(nn.Module):
    """Simpler Conv2D LOB model."""
    
    def __init__(self, gru_hidden=48, gru_layers=1, dropout=0.15):
        super().__init__()
        
        self.diff_gen = DifferenceFeatureGenerator()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 4), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=(1, 14)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        
        self.temporal_conv = nn.Sequential(
            CausalConv2dBlock(16, 24, kernel_size=(3, 1)),
            CausalConv2dBlock(24, 24, kernel_size=(3, 1)),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=24,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        
        self.head_t0 = nn.Linear(gru_hidden, 1)
        self.head_t1 = nn.Linear(gru_hidden, 1)
        
        self.gru_hidden = gru_hidden
        
    def reset_hidden(self):
        """Reset GRU and difference generator state for new sequence."""
        self.diff_gen.reset()
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        x = self.diff_gen(x)
        x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = self.temporal_conv(x)
        
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.dropout(x)
        
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.gru_hidden, device=x.device)
        
        gru_out, hidden = self.gru(x, hidden)
        last_hidden = gru_out[:, -1, :]
        
        pred_t0 = self.head_t0(last_hidden).squeeze(-1)
        pred_t1 = self.head_t1(last_hidden).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


# ============================================================================
# Prediction Model for Inference
# ============================================================================

class PredictionModel:
    """
    Conv2D LOB Solution (Exp 19).
    Single-step inference with hidden states carried across calls.
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "..", "best_model.pth")
        
        self.device = torch.device("cpu")
        
        self.model = Conv2DLOBSimple(
            gru_hidden=48,
            gru_layers=1,
            dropout=0.0,  # No dropout during inference
        )
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Conv2D LOB model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Make prediction for a single data point."""
        
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
            if self.model is not None:
                self.model.reset_hidden()
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        x = torch.tensor(data_point.state, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
        
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
            t0 = pred_t0.item()
            t1 = pred_t1.item()
        
        if not data_point.need_prediction:
            return None
        
        return np.array([t0, t1], dtype=np.float32)


# For local testing
if __name__ == "__main__":
    print("Testing Conv2DLOBSimple model...")
    
    # Test model forward pass (batch mode)
    model = Conv2DLOBSimple()
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    print(f"Batch mode - t0: {pred_t0.shape}, t1: {pred_t1.shape}")
    
    # Test single-step inference (need eval mode for batch norm)
    print("\nTesting single-step inference...")
    model.reset_hidden()
    model.eval()  # Critical for batch norm with batch_size=1
    hidden = None
    with torch.no_grad():
        for i in range(10):
            x = torch.randn(1, 1, 32)
            pred_t0, pred_t1, hidden = model(x, hidden)
            print(f"Step {i}: t0={pred_t0.item():.4f}, t1={pred_t1.item():.4f}")
    
    print("\nTesting PredictionModel wrapper...")
    pred_model = PredictionModel()
    
    for i in range(105):
        dp = DataPoint(
            seq_ix=0,
            step_in_seq=i,
            need_prediction=(i >= 99),
            state=np.random.randn(32).astype(np.float32),
        )
        pred = pred_model.predict(dp)
        if pred is not None:
            print(f"Step {i}: pred = {pred}")
