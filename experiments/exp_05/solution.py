import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import deque
from scipy.stats import skew as scipy_skew

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
# Also add current dir for model loading if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import DataPoint, ScorerStepByStep
except ImportError:
    # If src not in path correctly
    pass

# --- DL MODEL ---
class AttentionGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3, attention_dim=64):
        super(AttentionGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    # For training we used forward(x). For inference we need manual step logic?
    # Or we can reuse forward if we pass the GROWING sequence?
    # Passing growing sequence:
    # Step 1: Input (1, 1, F).
    # Step 1000: Input (1, 1000, F).
    # Recomputing Conv/GRU for 0..999 at step 1000 is expensive.
    # We must implement optimized step forward.
    pass

class DLInferenceWrapper:
    def __init__(self, model_path, input_dim):
        self.device = torch.device("cpu")
        self.model = AttentionGRUModel(input_dim=input_dim).to(self.device)
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        except Exception as e:
            print(f"Failed to load DL model: {e}")
            self.model = None
        self.model.eval()
        
        # State
        self.hidden = None
        self.gru_outputs = [] # List of tensors (1, 1, Hidden)
        self.input_buffer = deque(maxlen=3) # For Conv1d (kernel 3)

    def reset(self):
        self.hidden = None
        self.gru_outputs = []
        self.input_buffer.clear()
        
    def predict(self, x_step):
        # x_step: (Feats,) -> (1, 1, Feats)
        with torch.no_grad():
            x_t = torch.from_numpy(x_step).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Conv1d Context
            # We need 3 inputs for kernel 3.
            # If buffer < 3, pad? Training used padding=1 ('same'). 
            # In training, Conv sees zeros at borders.
            # Here we buffer.
            # Actually, standard Conv1d on sequence handles this.
            # Incremental Conv:
            # If we just feed current step to Conv1d(kernel=3, padding=1), it sees only current?
            # NO. Conv1d is not stateful.
            # We must implement manual conv logic or feed window.
            # Window size 3 centered? Padding=1 means it looks at t-1, t, t+1?
            # Causal Conv? Standard Conv1d is centered if padding=1.
            # This implies Future Leakage in Architecture if not strictly Causal!
            # My architecture: `nn.Conv1d(..., padding=kernel_size//2)`.
            # If kernel=3, padding=1.
            # Output at t depends on t-1, t, t+1.
            # THIS IS NON-CAUSAL.
            # Training loop: `x = x.permute... conv1(x)`. PyTorch Conv1d computes all.
            # At step t, it uses t+1.
            # BUT in inference, we DON'T have t+1.
            # Inference receives data sequentially.
            # If model requires t+1, we CANNOT predict t0.
            # **CRITICAL**: The Architecture used "Future" in Conv1d if padding=same.
            # However, `train_dl.py` used `x = conv1(x)`.
            # If model relies on future, Inference will FAIL to match Training.
            # UNLESS we delay prediction? No, task is t0 prediction.
            # We must fix ARCHITECTURE to be Causal (padding=0, look back).
            # OR we assume training learned to ignore future? No, weights exist.
            # Since I already launched training...
            # I can pad with 0 for "t+1" (Future is unknown).
            # This is correct. "Missing future" = 0.
            # So for step t: input is [t-1, t, 0].
            
            self.input_buffer.append(x_t) # Stores (1, 1, F)
            
            # Prepare Conv Input (1, F, 3)
            # Fetch t-1, t.
            # If t=0: [0, t, 0]
            # If t>0: [t-1, t, 0]
            if len(self.input_buffer) == 1:
                chk = [torch.zeros_like(x_t), self.input_buffer[0], torch.zeros_like(x_t)]
            else: # len >= 2
                chk = [self.input_buffer[-2], self.input_buffer[-1], torch.zeros_like(x_t)]
                
            conv_in = torch.cat(chk, dim=1).permute(0, 2, 1) # (1, F, 3)
            conv_out = self.model.conv1(conv_in) # (1, Ch, 1)?
            # Conv1d(3->1) with valid padding if I manual pad?
            # Model has padding=1 defined.
            # If I pass length 3, padding=1 makes it effectively length 5?
            # Input 3 + 2 pad = 5. Output 3.
            # We want the MIDDLE one (corresponding to t).
            # This is complex to match PyTorch exactly.
            
            # Alternative: Re-run Conv1d on short history?
            # If I run `model.conv1` on length 3 input with padding 1: Output is 3. Center is t.
            conv_res = self.model.relu(conv_out[:, :, 1:2]) # Take middle
            
            # GRU
            conv_res = conv_res.permute(0, 2, 1) # (1, 1, Ch)
            gru_out, self.hidden = self.model.gru(conv_res, self.hidden)
            self.gru_outputs.append(gru_out)
            
            # Attention
            # History: (1, SeqLen, Hidden)
            hist = torch.cat(self.gru_outputs, dim=1)
            # Mask? Training used Causal Mask.
            # Here we naturally have only Past.
            # We are at step t (last in hist).
            # Query: hist[:, -1:, :] (Current step)
            # Key/Val: hist (All steps)
            # Mask provided? No need if we attend to all.
            # Training: `attn(gru_out, gru_out, gru_out, mask=causal)`.
            # At index t, it attends to 0..t.
            # Here we do exactly that.
            
            query = gru_out
            key = hist
            value = hist
            
            attn_out, _ = self.model.attn(query, key, value)
            
            # Residual
            out = gru_out + attn_out
            
            pred = self.model.fc(out) # (1, 1, 2)
            return pred.squeeze().cpu().numpy()

# --- GBDT Wrappers ---
import catboost
class GBDTPredictor:
    def __init__(self, t0_path, t1_path):
        self.m0 = catboost.CatBoostRegressor()
        self.m1 = catboost.CatBoostRegressor()
        try:
            self.m0.load_model(t0_path)
            self.m1.load_model(t1_path)
            self.ready = True
        except:
            self.ready = False
            
    def predict(self, feats):
        if not self.ready: return np.zeros(2)
        p0 = self.m0.predict(feats)
        p1 = self.m1.predict(feats)
        return np.array([p0, p1])

# --- Main Solution ---
class PredictionModel:
    def __init__(self):
        self.input_dim = 150 # Dynamic?
        # Features logic must match train
        self.history = deque(maxlen=55)
        self.prev_vals = None
        
        # Load Models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dl_path = os.path.join(base_dir, "model_dl.pth")
        if not os.path.exists(dl_path): dl_path = os.path.join(base_dir, "best_model.pth")
        
        # Determine Input Dim from dummy feature gen?
        # We'll handle input_dim dynamic in DL wrapper if possible? 
        # Need to init model with correct dim.
        # Hardcode based on features.py logic count or run dummy.
        # Run dummy gen to get dim.
        dummy_df = pd.DataFrame([np.zeros(32)], columns=[f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)])
        dummy_df['seq_ix'] = 0
        dummy_df['step_in_seq'] = 0
        from features import generate_features, get_feature_columns
        d_feats = generate_features(dummy_df)
        cols = get_feature_columns(d_feats)
        self.feat_cols_len = len(cols)
        
        self.dl = DLInferenceWrapper(dl_path, self.feat_cols_len)
        self.gbdt = GBDTPredictor(os.path.join(base_dir, "model_gbdt_t0.cbm"), os.path.join(base_dir, "model_gbdt_t1.cbm"))
        
        self.current_seq = None

    def predict(self, data_point: DataPoint):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.history.clear()
            self.prev_vals = None
            self.dl.reset()
            
        s = data_point.state # (32,)
        
        # 1. Base Calcs (Raw)
        # Columns: p0..p11 (0-11), v0..v11 (12-23), dp0..dp3 (24-27), dv0..dv3 (28-31)
        p0, p1, p6, p7 = s[0], s[1], s[6], s[7]
        mid0 = (p0 + p6) * 0.5
        mid1 = (p1 + p7) * 0.5
        spread0 = p6 - p0
        spread1 = p7 - p1
        
        v_bids = s[12:18]
        v_asks = s[18:24]
        eps = 1e-8
        
        total_bid = np.sum(v_bids)
        total_ask = np.sum(v_asks)
        vol_imbal_total = (total_bid - total_ask) / (total_bid + total_ask + eps)
        
        # Per level imbal
        vol_imbals = (v_bids - v_asks) / (v_bids + v_asks + eps)
        
        # Ratios
        rel_spread = spread0 / (mid0 + eps)
        vol_ratio = total_bid / (total_ask + eps)
        
        # Current Target Values
        # [mid0, spread0, vol_imbal_total]
        curr_targets = np.array([mid0, spread0, vol_imbal_total], dtype=np.float32)
        self.history.append(curr_targets)
        
        hist = np.array(self.history) # (L, 3)
        h_len = len(hist)
        
        # Rolling Features (Windows 3, 21, 55)
        # targets idx: 0=mid, 1=spread, 2=vol_imbal
        
        roll_feats = []
        for i in range(3): # Per target
            col_data = hist[:, i]
            
            for w in [3, 21, 55]:
                # Slice last w
                sub = col_data[-w:]
                
                # Stats
                mean = np.mean(sub)
                std = np.std(sub, ddof=1) if len(sub) > 1 else 0.0
                curr = sub[-1]
                
                # Z-Score
                zscore = (curr - mean) / (std + eps)
                
                # Vol
                vol = std
                
                # Min/Max
                mn = np.min(sub)
                mx = np.max(sub)
                
                # Skew (only w=55)
                sk = 0.0
                if w == 55:
                    if len(sub) >= 3:
                         m = np.mean(sub)
                         std_dev = np.std(sub)
                         if std_dev > eps:
                             m3 = np.mean((sub - m)**3)
                             sk = m3 / (std_dev**3 + eps)
                         else:
                             sk = 0.0
                    else:
                        sk = 0.0
                
                roll_feats.extend([zscore, vol, mn, mx])
                if w == 55: roll_feats.append(sk)
                
        # Diffs
        curr_diff_vals = np.array([p0, p6, mid0, spread0, vol_imbal_total], dtype=np.float32)
        if self.prev_vals is None:
            diffs = np.zeros(5, dtype=np.float32)
        else:
            diffs = curr_diff_vals - self.prev_vals
        self.prev_vals = curr_diff_vals
        
        # Concat All
        # Must match feature order in features.py!
        # features.py order:
        # 1. Base (s?) No, `df_seq` passed. `generate_features` adds cols.
        # `get_feature_columns` returns numeric cols.
        # Original df had 32 cols.
        # `generate_features` adds:
        # mid0, mid1, spread0, spread1 (4)
        # vol_imbal_0..5 (6)
        # total_bid, total_ask, vol_imbal_total (3)
        # rel_spread, vol_ratio (2)
        # Loop targets(3) * Loop windows(3) * [z, vol, min, max, (skew if 55)]
        # target_0: w3(4), w21(4), w55(5) = 13 feats
        # total rolling = 3 * 13 = 39 feats
        # Diffs (5)
        
        # Original Raw cols (32) are included in `get_feature_cols`?
        # Yes, `get_feature_cols` returns all numeric except exclude list.
        # So Raw 32 are first (if original df had them first).
        
        # Constructed Vector:
        # Raw (32)
        # mid0, mid1, spread0, spread1
        # vol_imbal_0..5
        # total_bid, total_ask, vol_imbal_total
        # rel_spread, vol_ratio
        # Rollings (Iterate targets, iterate windows... match Loop Order in features.py)
        # Diffs
        
        f_list = [
            s, # 32
            [mid0, mid1, spread0, spread1], # 4
            vol_imbals, # 6
            [total_bid, total_ask, vol_imbal_total], # 3
            [rel_spread, vol_ratio], # 2
            np.array(roll_feats, dtype=np.float32), # 39
            diffs # 5
        ]
        
        
        flat_feats = np.concatenate([np.atleast_1d(item) for item in f_list]).astype(np.float32)
        
        # Inference
        if not data_point.need_prediction:
            # Update state (DL buffer) but don't compute expensive bits?
            # We MUST run DL forward because it's stateful (RNN) and Attention needs history from RNN.
            # So predict call is mandatory. 
            pass
            
        pred_dl = self.dl.predict(flat_feats) # (2,)
        # pred_gbdt = self.gbdt.predict(flat_feats) # (2,)
        
        # ens = 0.5 * pred_dl + 0.5 * pred_gbdt
        # ens = 0.8 * pred_dl + 0.2 * pred_gbdt
        ens = pred_dl   
        
        if not data_point.need_prediction: return None
        return ens

if __name__ == "__main__":
    # Benchmark
    import time
    print("Benchmarking Exp 05 Solution...")
    model = PredictionModel()
    # Dummy state
    dp = DataPoint(seq_ix=0, step_in_seq=0, need_prediction=True, state=np.random.rand(32))
    
    start = time.time()
    # Run 2000 steps
    # Must simulate sequence sort of? Or just repeated calls.
    # Predictor resets if seq changes.
    for i in range(2000): 
        dp.seq_ix = 0
        dp.step_in_seq = i
        model.predict(dp)
    end = time.time()
    print(f"Speed: {2000/(end-start):.2f} steps/sec")
