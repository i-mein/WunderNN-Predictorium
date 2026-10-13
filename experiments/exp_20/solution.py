"""
Exp 20: Ensemble Solution (exp_01 + exp_13)

Combines predictions from:
- exp_01: Simple GRU (LB 0.2820) - best generalization
- exp_13: Hybrid Transformer-GRU (LB 0.2262) - more sophisticated

Strategy: Weighted average of predictions.
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
# Model 1: Simple GRU from exp_01
# ============================================================================

class GRUModel(nn.Module):
    """Simple GRU model from exp_01."""
    
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        predictions = self.fc(out)
        return predictions, hidden


# ============================================================================
# Model 2: Hybrid Transformer-GRU from exp_13
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
    
    def forward_step(self, x, step):
        """Single step forward for inference."""
        x = x + self.pe[:, step:step+1, :]
        return x


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
    
    def forward_step(self, x, kv_cache=None, step=0):
        """Single step forward with KV cache for inference."""
        batch_size = x.shape[0]
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, 1, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        
        return self.out_proj(context), new_kv_cache


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
    
    def forward_step(self, x, kv_cache=None, step=0):
        attn_out, new_kv_cache = self.attn.forward_step(self.norm1(x), kv_cache, step)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv_cache


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
        
        # t1 head - deeper (must match saved model exactly)
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
        
        self.d_model = d_model
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.n_transformer_layers = n_transformer_layers
    
    def forward_step(self, x, hidden=None, kv_caches=None, step=0):
        """Single step forward for inference."""
        x = self.feature_engineer(x)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_encoder.forward_step(x, step)
        
        new_kv_caches = []
        for i, block in enumerate(self.transformer_blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block.forward_step(x, cache, step)
            new_kv_caches.append(new_cache)
        
        gru_out, new_gru_hidden = self.gru(x, hidden)
        
        combined = torch.cat([x, gru_out], dim=-1)
        
        pred_t0 = self.t0_head(combined).squeeze(-1).squeeze(-1)
        pred_t1 = self.t1_head(combined).squeeze(-1).squeeze(-1)
        
        return pred_t0, pred_t1, new_gru_hidden, new_kv_caches


# ============================================================================
# Ensemble Prediction Model
# ============================================================================

class PredictionModel:
    """
    Ensemble Solution combining exp_01 (Simple GRU) and exp_13 (Hybrid Transformer-GRU).
    
    Weighted average of predictions from both models.
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        
        # State for model 1 (GRU)
        self.hidden_gru = None
        
        # State for model 2 (Hybrid)
        self.hidden_hybrid = None
        self.kv_caches = None
        self.step = 0
        
        # Ensemble weights (exp_01 had better LB so give it more weight)
        self.weight_exp01 = 0.55  # exp_01 LB 0.2820
        self.weight_exp13 = 0.45  # exp_13 LB 0.2262
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cpu")
        
        # Load Model 1: exp_01 GRU
        model1_path = os.path.join(base_dir, "exp_01_model.pth")
        self.model1 = GRUModel(input_dim=32, hidden_dim=256, num_layers=2, output_dim=2)
        
        try:
            state_dict = torch.load(model1_path, map_location=self.device, weights_only=False)
            self.model1.load_state_dict(state_dict)
            self.model1.to(self.device)
            self.model1.eval()
            print(f"Loaded exp_01 GRU model from {model1_path}")
        except Exception as e:
            print(f"Error loading exp_01 model: {e}")
            self.model1 = None
        
        # Load Model 2: exp_13 Hybrid
        model2_path = os.path.join(base_dir, "exp_13_model.pth")
        self.model2 = HybridTransformerGRU(
            input_dim=32,
            d_model=128,
            n_heads=4,
            n_transformer_layers=2,
            gru_hidden=128,
            gru_layers=2,
            dropout=0.2,
        )
        
        try:
            checkpoint = torch.load(model2_path, map_location=self.device, weights_only=False)
            # exp_13 uses checkpoint format with model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model2.load_state_dict(state_dict)
            self.model2.to(self.device)
            self.model2.eval()
            print(f"Loaded exp_13 Hybrid model from {model2_path}")
        except Exception as e:
            print(f"Error loading exp_13 model: {e}")
            self.model2 = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_gru = None
            self.hidden_hybrid = None
            self.kv_caches = None
            self.step = 0

        state_tensor = torch.tensor(
            data_point.state, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        
        predictions = []
        weights = []
        
        # Model 1: exp_01 GRU
        if self.model1 is not None:
            with torch.no_grad():
                output, self.hidden_gru = self.model1(state_tensor, self.hidden_gru)
                pred1 = output.squeeze().numpy()  # [t0, t1]
                predictions.append(pred1)
                weights.append(self.weight_exp01)
        
        # Model 2: exp_13 Hybrid
        if self.model2 is not None:
            with torch.no_grad():
                pred_t0, pred_t1, self.hidden_hybrid, self.kv_caches = self.model2.forward_step(
                    state_tensor, self.hidden_hybrid, self.kv_caches, self.step
                )
                pred2 = np.array([pred_t0.item(), pred_t1.item()])
                predictions.append(pred2)
                weights.append(self.weight_exp13)
        
        self.step += 1
        
        if not data_point.need_prediction:
            return None
        
        if len(predictions) == 0:
            return np.zeros(2)
        
        # Weighted ensemble
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.zeros(2)
        for pred, w in zip(predictions, weights):
            ensemble_pred += w * pred
        
        return ensemble_pred
