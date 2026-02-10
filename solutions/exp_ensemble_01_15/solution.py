"""
Ensemble Solution: exp_01 (DL) + exp_15 (CatBoost)

Combines:
- exp_01: PyTorch GRU with raw features (LB: 0.2820)
- exp_15: CatBoost with 100 enhanced features (Val: 0.1980)

Ensemble Strategy: Weighted average with tunable weights.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from dataclasses import dataclass


# ============================================================
# DataPoint class (for compatibility)
# ============================================================
@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


# ============================================================
# exp_01: PyTorch GRU Model
# ============================================================
class GRUModel(nn.Module):
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


class DLPredictor:
    """exp_01 DL predictor wrapper."""
    
    def __init__(self, model_path):
        self.hidden = None
        self.current_seq_ix = None
        self.device = torch.device("cpu")
        
        self.model = GRUModel(input_dim=32, hidden_dim=256, num_layers=2, output_dim=2)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def reset(self):
        """Reset hidden state for new sequence."""
        self.hidden = None
        self.current_seq_ix = None
    
    def predict(self, step_data, seq_ix):
        """Predict using DL model."""
        # Reset on new sequence
        if self.current_seq_ix != seq_ix:
            self.current_seq_ix = seq_ix
            self.hidden = None
        
        # Prepare input: (1, 1, 32) - only use first 32 raw features
        state_tensor = torch.tensor(step_data[:32], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output, self.hidden = self.model(state_tensor, self.hidden)
        
        return output.squeeze().numpy()


# ============================================================
# exp_15: CatBoost Predictor (enhanced features)
# ============================================================
class CatBoostPredictor:
    """exp_15 CatBoost predictor wrapper with enhanced features."""
    
    def __init__(self, model_t0_path, model_t1_path):
        self.model_t0 = CatBoostRegressor()
        self.model_t1 = CatBoostRegressor()
        self.model_t0.load_model(model_t0_path)
        self.model_t1.load_model(model_t1_path)
        
        self.epsilon = 1e-8
        self.windows = [3, 10]
        self.reset()
    
    def reset(self):
        """Reset state for new sequence."""
        self.step = 0
        self.mid_price_0_buffer = []
        self.spread_0_buffer = []
        self.vol_imbal_total_buffer = []
        self.micro_price_deviation_buffer = []
        self.depth_imbal_near_buffer = []
        self.prev_values = {}
    
    def _compute_rolling_stats(self, buffer, windows):
        """Compute rolling mean and std."""
        stats = []
        arr = np.array(buffer)
        for w in windows:
            if len(arr) >= 1:
                window_data = arr[-w:] if len(arr) >= w else arr
                stats.append(np.mean(window_data))
                stats.append(np.std(window_data) if len(window_data) > 1 else 0.0)
            else:
                stats.extend([0.0, 0.0])
        return stats
    
    def generate_features(self, step_data):
        """Generate 100 features for CatBoost (matches exp_15)."""
        p = step_data[:12]   # Prices
        v = step_data[12:24] # Volumes
        
        features = []
        
        # 1. RAW FEATURES (24)
        features.extend(p)
        features.extend(v)
        
        # 2. BASE ENGINEERED FEATURES (16)
        mid_price_0 = (p[0] + p[6]) * 0.5
        mid_price_1 = (p[1] + p[7]) * 0.5
        features.append(mid_price_0)
        features.append(mid_price_1)
        
        spread_0 = p[6] - p[0]
        spread_1 = p[7] - p[1]
        features.append(spread_0)
        features.append(spread_1)
        
        for i in range(6):
            bv = v[i]
            av = v[i + 6]
            vol_imbal = (bv - av) / (bv + av + self.epsilon)
            features.append(vol_imbal)
        
        total_bid_vol = sum(v[:6])
        total_ask_vol = sum(v[6:])
        features.append(total_bid_vol)
        features.append(total_ask_vol)
        
        vol_imbal_total = (total_bid_vol - total_ask_vol) / \
                          (total_bid_vol + total_ask_vol + self.epsilon)
        features.append(vol_imbal_total)
        
        relative_spread_0 = spread_0 / (mid_price_0 + self.epsilon)
        features.append(relative_spread_0)
        
        vol_ratio = total_bid_vol / (total_ask_vol + self.epsilon)
        features.append(vol_ratio)
        
        price_depth = (p[5] - p[0]) / (p[0] + self.epsilon)
        features.append(price_depth)
        
        # 3. MICRO-PRICE FEATURES (5)
        micro_price = (p[0] * v[6] + p[6] * v[0]) / (v[0] + v[6] + self.epsilon)
        features.append(micro_price)
        
        micro_price_deviation = micro_price - mid_price_0
        features.append(micro_price_deviation)
        
        micro_price_deviation_norm = micro_price_deviation / (spread_0 + self.epsilon)
        features.append(micro_price_deviation_norm)
        
        weighted_mid_2 = (p[1] * v[7] + p[7] * v[1]) / (v[1] + v[7] + self.epsilon)
        features.append(weighted_mid_2)
        
        micro_price_deviation_2 = weighted_mid_2 - mid_price_1
        features.append(micro_price_deviation_2)
        
        # 4. ORDER BOOK PRESSURE (12)
        bid_pressure_0 = v[0] / (total_bid_vol + self.epsilon)
        bid_pressure_01 = (v[0] + v[1]) / (total_bid_vol + self.epsilon)
        ask_pressure_0 = v[6] / (total_ask_vol + self.epsilon)
        ask_pressure_01 = (v[6] + v[7]) / (total_ask_vol + self.epsilon)
        features.extend([bid_pressure_0, bid_pressure_01, ask_pressure_0, ask_pressure_01])
        
        bid_vwap = sum(p[i] * v[i] for i in range(6)) / (total_bid_vol + self.epsilon)
        ask_vwap = sum(p[i+6] * v[i+6] for i in range(6)) / (total_ask_vol + self.epsilon)
        
        bid_slope = (p[0] - bid_vwap) / (total_bid_vol + self.epsilon)
        ask_slope = (ask_vwap - p[6]) / (total_ask_vol + self.epsilon)
        slope_imbal = (bid_slope - ask_slope) / (bid_slope + ask_slope + self.epsilon)
        features.extend([bid_slope, ask_slope, slope_imbal])
        
        depth_imbal_near = (v[0] + v[1] - v[6] - v[7]) / \
                           (v[0] + v[1] + v[6] + v[7] + self.epsilon)
        depth_imbal_far = (v[4] + v[5] - v[10] - v[11]) / \
                          (v[4] + v[5] + v[10] + v[11] + self.epsilon)
        features.extend([depth_imbal_near, depth_imbal_far])
        
        features.extend([bid_vwap, ask_vwap, ask_vwap - bid_vwap])
        
        # 5. PRICE LEVEL FEATURES (6)
        price_range = p[11] - p[0]
        bid_range = p[5] - p[0]
        ask_range = p[11] - p[6]
        bid_spacing = (p[0] - p[5]) / 5
        ask_spacing = (p[11] - p[6]) / 5
        spacing_ratio = bid_spacing / (ask_spacing + self.epsilon)
        features.extend([price_range, bid_range, ask_range, bid_spacing, ask_spacing, spacing_ratio])
        
        # 6. ROLLING FEATURES (20)
        self.mid_price_0_buffer.append(mid_price_0)
        self.spread_0_buffer.append(spread_0)
        self.vol_imbal_total_buffer.append(vol_imbal_total)
        self.micro_price_deviation_buffer.append(micro_price_deviation)
        self.depth_imbal_near_buffer.append(depth_imbal_near)
        
        features.extend(self._compute_rolling_stats(self.mid_price_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.spread_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.vol_imbal_total_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.micro_price_deviation_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.depth_imbal_near_buffer, self.windows))
        
        # 7. DIFF FEATURES (8)
        diff_keys = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total',
                     'micro_price', 'micro_price_deviation', 'depth_imbal_near']
        diff_values = [p[0], p[6], mid_price_0, spread_0, vol_imbal_total,
                       micro_price, micro_price_deviation, depth_imbal_near]
        
        if self.step == 0:
            features.extend([0.0] * len(diff_keys))
        else:
            for key, val in zip(diff_keys, diff_values):
                features.append(val - self.prev_values.get(key, val))
        
        for key, val in zip(diff_keys, diff_values):
            self.prev_values[key] = val
        
        self.step += 1
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, step_data):
        """Predict using CatBoost models."""
        features = self.generate_features(step_data)
        features_2d = features.reshape(1, -1)
        
        t0_pred = self.model_t0.predict(features_2d)[0]
        t1_pred = self.model_t1.predict(features_2d)[0]
        
        return np.array([t0_pred, t1_pred], dtype=np.float32)


# ============================================================
# Ensemble Model
# ============================================================
class EnsembleModel:
    """
    Ensemble of exp_01 (DL) and exp_15 (CatBoost).
    
    Weights can be tuned based on validation performance.
    """
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load DL model (exp_01)
        dl_model_path = os.path.join(base_dir, "best_model.pth")
        self.dl_predictor = DLPredictor(dl_model_path)
        
        # Load CatBoost models (exp_15)
        cb_t0_path = os.path.join(base_dir, "model_catboost_t0.cbm")
        cb_t1_path = os.path.join(base_dir, "model_catboost_t1.cbm")
        self.cb_predictor = CatBoostPredictor(cb_t0_path, cb_t1_path)
        
        # Ensemble weights: [DL_weight, CatBoost_weight] for [t0, t1]
        # Based on relative performance:
        # - exp_01: LB 0.2820 (better generalization)
        # - exp_15: Val 0.1980 (t0: 0.3130, t1: 0.0830)
        # DL seems better on LB, so give it more weight
        self.weights_t0 = [0.6, 0.4]  # DL, CatBoost for t0
        self.weights_t1 = [0.7, 0.3]  # DL, CatBoost for t1 (DL seems better)
        
        self.current_seq_ix = None
    
    def reset(self):
        """Reset state for new sequence."""
        self.dl_predictor.reset()
        self.cb_predictor.reset()
        self.current_seq_ix = None
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Ensemble prediction combining DL and CatBoost.
        """
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.dl_predictor.reset()
            self.cb_predictor.reset()
        
        # Get predictions from both models
        dl_pred = self.dl_predictor.predict(data_point.state, data_point.seq_ix)
        cb_pred = self.cb_predictor.predict(data_point.state)
        
        if not data_point.need_prediction:
            return None
        
        # Weighted ensemble
        t0_pred = self.weights_t0[0] * dl_pred[0] + self.weights_t0[1] * cb_pred[0]
        t1_pred = self.weights_t1[0] * dl_pred[1] + self.weights_t1[1] * cb_pred[1]
        
        return np.array([t0_pred, t1_pred], dtype=np.float32)


# ============================================================
# Submission Interface
# ============================================================
class PredictionModel:
    """Main prediction model class for submission."""
    
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ensemble = EnsembleModel(base_dir)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        return self.ensemble.predict(data_point)


# ============================================================
# Testing
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(base_dir, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Loading validation data...")
        df = pd.read_parquet(test_file)
        
        model = PredictionModel()
        
        predictions = []
        targets = []
        
        print("Running ensemble predictions...")
        for row in tqdm(df.values):
            seq_ix = int(row[0])
            step_in_seq = int(row[1])
            need_prediction = bool(row[2])
            lob_data = row[3:35].astype(np.float32)
            labels = row[35:37]
            
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            pred = model.predict(data_point)
            
            if need_prediction and pred is not None:
                predictions.append(pred)
                targets.append(labels)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate Pearson correlation
        from scipy.stats import pearsonr
        
        corr_t0, _ = pearsonr(targets[:, 0], predictions[:, 0])
        corr_t1, _ = pearsonr(targets[:, 1], predictions[:, 1])
        avg_corr = (corr_t0 + corr_t1) / 2
        
        print(f"\nEnsemble Results:")
        print(f"  t0 Pearson: {corr_t0:.4f}")
        print(f"  t1 Pearson: {corr_t1:.4f}")
        print(f"  Avg Pearson: {avg_corr:.4f}")
    else:
        print(f"Test file not found: {test_file}")
