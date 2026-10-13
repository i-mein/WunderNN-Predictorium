"""
Exp 13: Hybrid Transformer-GRU Solution

Solution file for submission - standalone single-step inference.
Fixed to use single-step processing like exp_10 for speed.
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

class FeatureEngineer(nn.Module):
    """Computes engineered features."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        p0, p6 = prices[..., 0:1], prices[..., 6:7]
        v0, v6 = volumes[..., 0:1], volumes[..., 6:7]
        
        mid_price = (p0 + p6) / 2
        spread = p6 - p0
        
        bid_vol = volumes[..., :6]
        ask_vol = volumes[..., 6:]
        vol_imbalance = (bid_vol.sum(dim=-1, keepdim=True) - ask_vol.sum(dim=-1, keepdim=True)) / \
                        (bid_vol.sum(dim=-1, keepdim=True) + ask_vol.sum(dim=-1, keepdim=True) + 1e-8)
        
        best_vol_imbalance = (v0 - v6) / (v0 + v6 + 1e-8)
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        weighted_bid = (prices[..., :6] * volumes[..., :6]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., :6].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_ask = (prices[..., 6:] * volumes[..., 6:]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., 6:].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_mid = (weighted_bid + weighted_ask) / 2
        
        engineered = torch.cat([
            mid_price, spread, vol_imbalance, best_vol_imbalance,
            price_momentum, vol_momentum, weighted_mid,
        ], dim=-1)
        
        return torch.cat([x, engineered], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.step = 0  # Track current step for inference
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    def forward_step(self, x, step):
        """Single step forward for inference."""
        x = x + self.pe[:, step:step+1, :]
        return x  # No dropout during inference


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class HybridTransformerGRU(nn.Module):
    def __init__(
        self,
        input_dim=32,
        d_model=128,
        n_heads=4,
        n_transformer_layers=2,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        
        self.feature_engineer = FeatureEngineer()
        engineered_dim = input_dim + 7
        
        self.input_proj = nn.Linear(engineered_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_transformer_layers)
        ])
        
        self.gru = nn.GRU(
            d_model,
            gru_hidden,
            gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False,
        )
        
        combined_dim = d_model + gru_hidden
        
        self.t0_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Delta features attention for t1 (must match saved model)
        self.delta_attn = nn.Sequential(
            nn.Linear(8, 32),  # 4 dp + 4 dv
            nn.GELU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.d_model = d_model
    
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape
        
        x_eng = self.feature_engineer(x)
        h = self.input_proj(x_eng)
        h = self.input_norm(h)
        h = self.pos_encoder(h)
        
        for block in self.transformer_blocks:
            h = block(h)
        
        transformer_out = h
        
        if hidden is None:
            hidden = torch.zeros(
                self.gru_layers, batch_size, self.gru_hidden,
                device=x.device
            )
        
        gru_out, hidden = self.gru(transformer_out, hidden)
        combined = torch.cat([transformer_out, gru_out], dim=-1)
        
        pred_t0 = self.t0_head(combined).squeeze(-1)
        pred_t1 = self.t1_head(combined).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


# ============================================================================
# Prediction Model for Inference - Single Step (like exp_10)
# ============================================================================

class PredictionModel:
    """
    Hybrid Transformer-GRU Solution (Exp 13).
    
    Single-step inference with GRU hidden state carried across calls.
    Transformer attention is over single step only (no history buffer).
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Determine model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "..", "best_model.pth")
        
        self.device = torch.device("cpu")
        
        # Initialize model
        self.model = HybridTransformerGRU(
            input_dim=32,
            d_model=128,
            n_heads=4,
            n_transformer_layers=2,
            gru_hidden=128,
            gru_layers=2,
            dropout=0.0,  # No dropout during inference
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Hybrid Transformer-GRU model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Make prediction for a single data point."""
        
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Prepare input: single step (1, 1, 32)
        x = torch.tensor(data_point.state, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
        
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
            
            # Extract single prediction
            t0 = pred_t0[0, 0].item()
            t1 = pred_t1[0, 0].item()
        
        if not data_point.need_prediction:
            return None
        
        return np.array([t0, t1], dtype=np.float32)


# For local testing
if __name__ == "__main__":
    model = PredictionModel()
    
    # Test with dummy data
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
