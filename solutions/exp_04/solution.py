import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
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
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        out, hidden = self.gru(x, hidden)
        outputs = self.fc(out)
        return outputs, hidden

class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        if not os.path.exists(model_path):
              model_path = os.path.join(base_dir, "..", "..", "best_model.pth") # Fallback

        self.device = torch.device("cpu")
        self.input_dim = 53
        
        self.model = CNNGRUModel(input_dim=self.input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.model = None

        # State: Rolling History of [mid0, spread0]
        # Max window needed is 20 (for BB). RSI uses 14.
        self.history_len = 20
        self.roll_buffer = deque(maxlen=self.history_len) # Stores [mid0, spread0]
        
        # State: Diffs previous values
        self.prev_vals = None # [p0, p6, mid0, spread0]

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None 
            self.prev_vals = None
            self.roll_buffer.clear()

        s = data_point.state # (32,)
        
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
        
        # Update Buffer [mid0, spread0]
        curr_targets = np.array([mid0, spread0], dtype=np.float32)
        self.roll_buffer.append(curr_targets)
        
        hist = np.array(self.roll_buffer) # (N, 2)
        
        # 2. RSI (14) - Cutler's
        # Need at least 2 points to get difference? min_periods=1 in Pandas handles this.
        # Diff of history.
        # If len=1, diff is empty? Pandas rolling mean of empty is NaN. fillna(0)?
        # Our `features.py` uses `grouped.diff().fillna(0)`.
        # So first step diff is 0.
        
        rsi_feats = []
        for i in range(2): # mid0, spread0
            h = hist[:, i]
            if len(h) < 2:
                # Delta 0
                rsi = 50.0 # Neutral? Or based on fillna(0) delta -> gain=0, loss=0 -> rs=nan -> 50?
                # Formula: 100 - 100/(1+RS). RS = AvgGain/AvgLoss.
                # If loss=0, RS=inf -> RSI=100.
                # If gain=0, RS=0 -> RSI=0.
                # If both 0 (flat), usually RSI=50.
                rsi_feats.append(50.0)
            else:
                # Calculate diffs
                # We need diffs aligned with history. 
                # Diff[0] = 0 (assumed by fillna(0) logic in features.py usually, 
                # but Pandas rolling logic on diff series implies we look at window of diffs).
                # `diffs = grouped.diff().fillna(0)` creates a series of diffs.
                # `temp.groupby...rolling(14).mean()` takes mean of last 14 diffs.
                
                # So we construct full diff array from history.
                # Prepend 0 logic? Or just diff. 
                d = np.diff(h) 
                # Only N-1 diffs.
                # If we want to match Pandas window 14 on "diffs series", we need last 14 diffs.
                # If N < 14, we pad or use what we have.
                # We should assume the first diff of sequence is 0.
                # So we concatenate [0] + d.
                d_full = np.concatenate(([0.0], d))
                
                # Take last 14
                w_d = d_full[-14:]
                
                g = np.maximum(w_d, 0)
                l = -np.minimum(w_d, 0)
                
                avg_g = np.mean(g)
                avg_l = np.mean(l)
                
                rs_val = avg_g / (avg_l + eps)
                rsi_val = 100.0 - (100.0 / (1.0 + rs_val))
                rsi_feats.append(rsi_val)
                
        # 3. Bollinger %B (20)
        # Uses last 20 prices.
        bb_feats = []
        for i in range(2):
            h = hist[:, i] # All available history (max 20)
            # Rolling window 20
            # If len < 20, use all (min_periods=1 match)
            
            ma = np.mean(h)
            std = np.std(h, ddof=1) if len(h) > 1 else 0.0
            
            price = h[-1] # Current price
            
            # %B = (Price - MA + 2Std) / 4Std
            pb = (price - ma + 2*std) / (4*std + eps)
            bb_feats.append(pb)
            
        # 4. Diffs (Standard)
        if self.prev_vals is None:
            diffs = np.zeros(4, dtype=np.float32)
        else:
            curr_vals_for_diff = np.array([p0, p6, mid0, spread0], dtype=np.float32)
            diffs = curr_vals_for_diff - self.prev_vals
            
        self.prev_vals = np.array([p0, p6, mid0, spread0], dtype=np.float32)
        
        # 5. Concat
        # Order Check:
        # [Raw 32]
        # [Phase 1 (13)]: mid0, mid1, spread0, spread1, imbals(6), total_bid, total_ask, total_imbal
        # [Phase 2 (4)]: RSI(mid0), RSI(spread0), BB(mid0), BB(spread0)  <-- Verify loop order
        # Loop over target_cols=['mid_price_0', 'spread_0'].
        # For col: calc RSI using temp df. 
        # features.py logic:
        # for col in targets:
        #    ... df[col_rsi] ...
        #    ... df[col_bb] ...
        # So order: RSI(mid), BB(mid), RSI(spread), BB(spread).
        # My Step 2/3 lists separate logic.
        # rsi_feats = [rsi_mid, rsi_spread]. bb_feats = [bb_mid, bb_spread].
        # I need to interleave them to match features.py!
        # features.py: 
        #   mid_rsi, mid_bb 
        #   spread_rsi, spread_bb
        
        tech_feats = []
        tech_feats.append(rsi_feats[0]) # mid rsi
        tech_feats.append(bb_feats[0])  # mid bb
        tech_feats.append(rsi_feats[1]) # spread rsi
        tech_feats.append(bb_feats[1])  # spread bb
        
        # [Phase 3 (4)]: Diffs
        
        features = np.concatenate([
            s,
            [mid0, mid1, spread0, spread1],
            vol_imbals,
            [total_bid, total_ask, total_imbal],
            np.array(tech_feats, dtype=np.float32),
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
    print("Benchmarking Exp 04 Solution...")
    
    model = PredictionModel()
    dp = DataPoint(seq_ix=0, step_in_seq=0, need_prediction=True, state=np.random.rand(32))
    
    start = time.time()
    for _ in range(2000):
        with torch.no_grad():
            model.predict(dp)
    end = time.time()
    print(f"Inference Speed: {2000/(end-start):.2f} steps/sec")
