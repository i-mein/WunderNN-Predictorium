import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from utils import DataPoint, ScorerStepByStep

# --- MODEL DEFINITION ---
class CNNGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (Batch, Seq, Feats)
        x = x.permute(0, 2, 1) # (Batch, Feats, Seq)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1) # (Batch, Seq, Channels)
        out, hidden = self.gru(x, hidden)
        outputs = self.fc(out)
        return outputs, hidden

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
        self.input_dim = 61
        
        self.model = CNNGRUModel(input_dim=self.input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # State for Rolling Features
        # We need to track: [mid_price_0, spread_0, vol_imbal_total]
        # Max window is 50.
        self.history_len = 50
        # Buffer to store last 50 values for the 3 rolling features
        # Shape: (50, 3)
        self.roll_buffer = deque(maxlen=self.history_len)
        
        # State for Diffs (p0, p6, mid0, spread0)
        self.prev_vals = None 

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None 
            self.prev_vals = None
            self.roll_buffer.clear()

        s = data_point.state # (32,)
        
        # --- Feature Engineering (NumPy) ---
        
        # 1. Base Calcs
        p0, p1, p6, p7 = s[0], s[1], s[6], s[7]
        v_bids = s[12:18]
        v_asks = s[18:24]
        eps = 1e-8
        
        mid0 = (p0 + p6) * 0.5
        mid1 = (p1 + p7) * 0.5
        spread0 = p6 - p0
        spread1 = p7 - p1
        
        vol_imbals = (v_bids - v_asks) / (v_bids + v_asks + eps)
        
        total_bid = np.sum(v_bids)
        total_ask = np.sum(v_asks)
        total_imbal = (total_bid - total_ask) / (total_bid + total_ask + eps)
        
        # Update Rolling Buffer with [mid0, spread0, total_imbal]
        current_roll_vals = np.array([mid0, spread0, total_imbal], dtype=np.float32)
        self.roll_buffer.append(current_roll_vals)
        
        # 2. Rolling Stats
        # Need efficient calc. Deque to array is small (50x3). Fast enough.
        # If history < window, use all available (min_periods=1)
        
        hist_arr = np.array(self.roll_buffer) # (N, 3)
        
        roll_feats = []
        for w in [10, 50]:
            # Slice last w
            if len(self.roll_buffer) >= w:
                window_data = hist_arr[-w:]
            else:
                window_data = hist_arr # Use all
                
            # Mean (Collapses to (3,))
            means = np.mean(window_data, axis=0)
            
            # Std (Collapses to (3,))
            # Pandas default ddof=1. NumPy default 0. User usually matches. 
            # Features.py used `rolling().std()`. Pandas defaults to ddof=1.
            if len(window_data) > 1:
                stds = np.std(window_data, axis=0, ddof=1)
            else:
                stds = np.zeros(3)
                
            # Order in features.py: Mean cols then Std cols.
            # features.py loop:
            # - Mean (mid, spread, imbal)
            # - Std (mid, spread, imbal)
            roll_feats.extend(means)
            roll_feats.extend(stds)
            
        # 3. Diffs
        if self.prev_vals is None:
            diffs = np.zeros(4, dtype=np.float32)
        else:
            curr_vals_for_diff = np.array([p0, p6, mid0, spread0], dtype=np.float32)
            diffs = curr_vals_for_diff - self.prev_vals
            
        self.prev_vals = np.array([p0, p6, mid0, spread0], dtype=np.float32)
        
        # 4. Concat
        # Order Check:
        # [Raw 32]
        # [Phase 1 (13)]: mid0, mid1, spread0, spread1, vol_imbals(6), total_bid, total_ask, total_imbal
        # [Phase 2 (12)]: Window 10 [Mean(3), Std(3)], Window 50 [Mean(3), Std(3)]
        # [Phase 3 (4)]: Diffs
        
        features = np.concatenate([
            s,
            [mid0, mid1, spread0, spread1],
            vol_imbals,
            [total_bid, total_ask, total_imbal],
            np.array(roll_feats, dtype=np.float32),
            diffs
        ])
        
        # Inference
        state_tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
        
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
    print("Benchmarking Exp 03 Hybrid Inference...")
    model = PredictionModel()
    dp = DataPoint(seq_ix=0, step_in_seq=0, need_prediction=True, state=np.random.rand(32))
    
    # Warmup
    for _ in range(100): model.predict(dp)
    
    start = time.time()
    n = 2000
    for i in range(n):
        dp.seq_ix = 0 if i < 1500 else 1
        model.predict(dp)
    end = time.time()
    
    print(f"Speed: {n / (end - start):.2f} steps/sec")
