"""
exp_23: TCN+GRU with Enhanced LOB Features

Submission-ready solution for step-by-step inference.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    from utils import DataPoint, ScorerStepByStep


# ============================================
# Feature Engineering (copied for standalone)
# ============================================
def safe_divide(a, b, default=0.0):
    """Safe division that handles zero denominators."""
    if np.abs(b) > 1e-10:
        return a / b
    return default


def safe_log(x, default=0.0):
    """Safe log that handles non-positive values."""
    if x > 1e-10:
        return np.log(x)
    return default


def compute_enhanced_features(state: np.ndarray) -> np.ndarray:
    """
    Compute enhanced LOB features from raw 32-feature state.
    Total: 47 features = 32 raw + 15 enhanced
    """
    # Handle NaN in input
    s = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract raw components
    p_bid = s[0:6]
    p_ask = s[6:12]
    v_bid = np.maximum(s[12:18], 0.0)
    v_ask = np.maximum(s[18:24], 0.0)
    dp = s[24:28]
    dv = s[28:32]
    
    best_bid = p_bid[0]
    best_ask = p_ask[0]
    best_bid_vol = max(v_bid[0], 0.0)
    best_ask_vol = max(v_ask[0], 0.0)
    
    # Enhanced Features (15)
    mid_price = (best_bid + best_ask) * 0.5
    
    total_best_vol = best_bid_vol + best_ask_vol
    if total_best_vol > 1e-10:
        micro_price = (best_ask * best_bid_vol + best_bid * best_ask_vol) / total_best_vol
    else:
        micro_price = mid_price
    
    spread = best_ask - best_bid
    relative_spread = safe_divide(spread, abs(mid_price) + 1e-8)
    
    vol_sum_0 = best_bid_vol + best_ask_vol
    vol_imbal_0 = safe_divide(best_bid_vol - best_ask_vol, vol_sum_0 + 1e-8)
    
    total_bid_vol = np.sum(v_bid)
    total_ask_vol = np.sum(v_ask)
    total_vol = total_bid_vol + total_ask_vol
    vol_imbal_total = safe_divide(total_bid_vol - total_ask_vol, total_vol + 1e-8)
    
    # Weighted Price Pressure
    wpp = 0.0
    weights = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    for i in range(6):
        vol_sum = v_bid[i] + v_ask[i]
        if vol_sum > 1e-10:
            level_imbal = (v_bid[i] - v_ask[i]) / vol_sum
            wpp += weights[i] * level_imbal
    
    # Order Flow Imbalance
    ofi = dv[0] - dv[1]
    
    price_momentum = np.sum(dp)
    vol_momentum = np.sum(dv)
    
    if total_ask_vol > 1e-10:
        depth_ratio = total_bid_vol / total_ask_vol
    else:
        depth_ratio = 1.0
    log_depth_ratio = safe_log(depth_ratio)
    
    bid_depth = safe_divide(best_bid - p_bid[5], abs(best_bid) + 1e-8)
    ask_depth = safe_divide(p_ask[5] - best_ask, abs(best_ask) + 1e-8)
    
    mid_micro_diff = micro_price - mid_price
    norm_spread = safe_divide(spread, abs(mid_price) + 1e-8)
    
    enhanced = np.array([
        mid_price, micro_price, spread, relative_spread,
        vol_imbal_0, vol_imbal_total, wpp, ofi,
        price_momentum, vol_momentum, log_depth_ratio,
        bid_depth, ask_depth, mid_micro_diff, norm_spread,
    ], dtype=np.float32)
    
    # Final NaN cleanup
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
    
    result = np.concatenate([s, enhanced])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result.astype(np.float32)


# ============================================
# Model Architecture (copied for standalone)
# ============================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        layers = []
        dilations = [2**i for i in range(num_layers)]
        for i, dilation in enumerate(dilations):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        return x


class TCNGRUModel(nn.Module):
    def __init__(self, input_dim=47, tcn_hidden=64, tcn_layers=4, gru_hidden=128, gru_layers=2, output_dim=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, tcn_hidden)
        self.tcn = TCN(input_dim=tcn_hidden, hidden_dim=tcn_hidden, kernel_size=3, num_layers=tcn_layers, dropout=dropout)
        self.gru = nn.GRU(input_size=tcn_hidden, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=False, dropout=dropout if gru_layers > 1 else 0)
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        x = self.tcn(x)
        x, hidden = self.gru(x, hidden)
        output = self.output_head(x)
        return output, hidden


# ============================================
# Prediction Model for Step-by-Step Inference
# ============================================
class PredictionModel:
    """
    exp_23: TCN+GRU with Enhanced LOB Features
    
    Step-by-step inference with stateful GRU.
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            model_path = os.path.join(base_dir, "best_model.pth")
        
        self.device = torch.device("cpu")
        
        # Model configuration
        self.input_dim = 47
        self.model = TCNGRUModel(
            input_dim=self.input_dim,
            tcn_hidden=64,
            tcn_layers=4,
            gru_hidden=128,
            gru_layers=2,
            dropout=0.2
        )
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Predict for a single data point.
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
        
        # Compute enhanced features
        features = compute_enhanced_features(data_point.state)
        
        # Prepare input tensor (1, 1, 47)
        x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Inference
        with torch.no_grad():
            output, self.hidden = self.model(x, self.hidden)
        
        if not data_point.need_prediction:
            return None
        
        prediction = output.squeeze().cpu().numpy()
        return prediction


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing exp_23 solution...")
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("\nRunning validation...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in ['t0', 't1']:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Validation file not found at {test_file}")
