import os
import sys
import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint, ScorerStepByStep

# Adjust path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- MODEL (Same) ---
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden

class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Load Model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        if not os.path.exists(model_path):
              model_path = os.path.join(base_dir, "..", "..", "best_model.pth")

        self.device = torch.device("cpu")
        self.input_dim = 49 
        self.model = GRUModel(input_dim=self.input_dim, hidden_dim=256, num_layers=2, output_dim=2)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.model = None
            
        # Optimization: Pre-allocate state for Diff calc
        # We need p0, p6, mid_price_0, spread_0 from previous step
        # raw data state indices:
        # p0..p5 (0..5), p6..p11 (6..11)
        # mid_price_0 = (p0 + p6) / 2
        # spread_0 = p6 - p0
        self.prev_vals = None # [p0, p6, mid0, spread0]

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None 
            self.prev_vals = None

        # --- FAST FEATURE GEN (NumPy) ---
        # Raw State: p0..p11 (0-11), v0..v11 (12-23), dp0..dp3 (24-27), dv0..dv3 (28-31)
        # Total 32 raw
        # Exp 02 Features Order from features.py:
        # 0-31: Raw (p, v, dp, dv) - Note: features.py might re-order if using columns.
        # Let's verify features.py order.
        # features.py logic:
        # df copy -> add mid_price_0, mid_price_1, spread_0, spread_1 ...
        # get_feature_columns excludes meta.
        # Order is: [Original 32 cols] + [New Cols]
        # Original 32 cols order in DataFrame read from parquet matches DataPoint.state order?
        # Parquet Cols: p0..p11, v0..v11, dp0..3, dv0..3.
        # DataPoint.state is also likely this order.
        # So we can just concat.
        
        s = data_point.state # Shape (32,)
        
        # 1. Engineered Features
        # Indices: p0=0, p1=1, p6=6, p7=7
        # v0..v5 = 12..17, v6..v11 = 18..23
        
        p0, p1, p6, p7 = s[0], s[1], s[6], s[7]
        
        mid0 = (p0 + p6) * 0.5
        mid1 = (p1 + p7) * 0.5
        spread0 = p6 - p0
        spread1 = p7 - p1
        
        # Vol Imbalance
        # v indices: 12+i (bid), 18+i (ask)
        eps = 1e-8
        v_bids = s[12:18]
        v_asks = s[18:24]
        
        # Vectorized vol imbal
        # (vb - va) / (vb + va + eps)
        vol_imbals = (v_bids - v_asks) / (v_bids + v_asks + eps) # Shape (6,)
        
        total_bid = np.sum(v_bids)
        total_ask = np.sum(v_asks)
        total_imbal = (total_bid - total_ask) / (total_bid + total_ask + eps)
        
        # Diffs
        # Cols to diff: p0, p6, mid_price_0, spread_0
        if self.prev_vals is None:
            # First step diff is 0
            diffs = np.zeros(4, dtype=np.float32)
        else:
            curr_vals_for_diff = np.array([p0, p6, mid0, spread0], dtype=np.float32)
            diffs = curr_vals_for_diff - self.prev_vals
            
        # Update state
        self.prev_vals = np.array([p0, p6, mid0, spread0], dtype=np.float32)
        
        # Concatenate all features
        # Order MUST match training:
        # [Original 32], [mid0, mid1, spread0, spread1], [vol_imbal 0..5], [total_bid, total_ask, total_imbal], [diffs 0..3]
        # Total: 32 + 4 + 6 + 3 + 4 = 49. Matches.
        
        features = np.concatenate([
            s, 
            [mid0, mid1, spread0, spread1],
            vol_imbals,
            [total_bid, total_ask, total_imbal],
            diffs
        ])
        
        # --- INFERENCE ---
        state_tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0) # (1, 1, 49)
        
        if self.model is None:
             return np.zeros(2) if data_point.need_prediction else None

        with torch.no_grad():
            output, self.hidden = self.model(state_tensor, self.hidden)
        
        if not data_point.need_prediction:
            return None
            
        return output.squeeze().numpy()

if __name__ == "__main__":
    # Benchmark
    import time
    print("Benchmarking Inference Speed...")
    model = PredictionModel()
    # Dummy DataPoint
    dp = DataPoint(seq_ix=0, step_in_seq=0, need_prediction=True, state=np.random.rand(32))
    
    # Warmup
    for _ in range(100): model.predict(dp)
    
    start = time.time()
    n = 10000
    for i in range(n):
        dp.seq_ix = 0 if i < 900 else 1 # Change seq sometimes
        model.predict(dp)
    end = time.time()
    
    print(f"Speed: {n / (end - start):.2f} steps/sec")
