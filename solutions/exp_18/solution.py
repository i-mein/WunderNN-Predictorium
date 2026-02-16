"""
Exp 18: Multi-Stream Hybrid Solution

Solution file for submission - standalone single-step inference.
"""

import os
import math
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

class LeakageFreeFeatures(nn.Module):
    """Compute engineered features from CURRENT STEP ONLY."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        p0 = prices[..., 0:1]
        p6 = prices[..., 6:7]
        v0 = volumes[..., 0:1]
        v6 = volumes[..., 6:7]
        
        mid_price = (p0 + p6) / 2
        spread = p6 - p0
        
        bid_vol = volumes[..., :6].sum(dim=-1, keepdim=True)
        ask_vol = volumes[..., 6:].sum(dim=-1, keepdim=True)
        vol_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        
        best_vol_imbalance = (v0 - v6) / (v0 + v6 + 1e-8)
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        engineered = torch.cat([
            mid_price, spread, vol_imbalance,
            best_vol_imbalance, price_momentum, vol_momentum,
        ], dim=-1)
        
        return engineered


class StreamEncoder(nn.Module):
    """Encode a single feature stream using projection + GRU."""
    
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
        h = self.input_proj(x)
        output, hidden = self.gru(h, hidden)
        return output, hidden


class MultiStreamHybrid(nn.Module):
    """Multi-stream architecture with separate encoders for each feature category."""
    
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
        
        self.feature_eng = LeakageFreeFeatures()
        
        self.price_stream = StreamEncoder(12, price_hidden, price_output, 2, dropout)
        self.volume_stream = StreamEncoder(12, volume_hidden, volume_output, 2, dropout)
        self.dprice_stream = StreamEncoder(4, delta_hidden, delta_output, 1, dropout)
        self.dvolume_stream = StreamEncoder(4, delta_hidden, delta_output, 1, dropout)
        
        self.eng_proj = nn.Sequential(
            nn.Linear(6, eng_output),
            nn.LayerNorm(eng_output),
            nn.GELU(),
        )
        
        combined_dim = price_output + volume_output + delta_output * 2 + eng_output
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.t0_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self.price_output = price_output
        self.volume_output = volume_output
        self.delta_output = delta_output
    
    def forward(self, x, hidden=None):
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        eng_features = self.feature_eng(x)
        
        if hidden is None:
            hidden = {
                'price': None,
                'volume': None,
                'dprice': None,
                'dvolume': None,
            }
        
        price_out, hidden['price'] = self.price_stream(prices, hidden['price'])
        volume_out, hidden['volume'] = self.volume_stream(volumes, hidden['volume'])
        dprice_out, hidden['dprice'] = self.dprice_stream(dprices, hidden['dprice'])
        dvolume_out, hidden['dvolume'] = self.dvolume_stream(dvolumes, hidden['dvolume'])
        
        eng_out = self.eng_proj(eng_features)
        
        combined = torch.cat([
            price_out, volume_out, dprice_out, dvolume_out, eng_out,
        ], dim=-1)
        
        fused = self.fusion(combined)
        
        pred_t0 = self.t0_head(fused).squeeze(-1)
        pred_t1 = self.t1_head(fused).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


# ============================================================================
# Prediction Model for Inference
# ============================================================================

class PredictionModel:
    """
    Multi-Stream Hybrid Solution (Exp 18).
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
        
        self.model = MultiStreamHybrid(
            price_hidden=48,
            price_output=64,
            volume_hidden=48,
            volume_output=64,
            delta_hidden=32,
            delta_output=32,
            eng_output=32,
            head_hidden=128,
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
            print(f"Loaded Multi-Stream Hybrid from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Make prediction for a single data point."""
        
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        x = torch.tensor(data_point.state, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
        
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
            t0 = pred_t0[0, 0].item()
            t1 = pred_t1[0, 0].item()
        
        if not data_point.need_prediction:
            return None
        
        return np.array([t0, t1], dtype=np.float32)


# For local testing
if __name__ == "__main__":
    model = PredictionModel()
    
    for i in range(105):
        dp = DataPoint(
            seq_ix=0,
            step_in_seq=i,
            need_prediction=(i >= 99),
            state=np.random.randn(32).astype(np.float32),
        )
        pred = model.predict(dp)
        if pred is not None:
            print(f"Step {i}: pred = {pred}")
