import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    from utils import DataPoint, ScorerStepByStep


# --- MODEL DEFINITION (Copied for standalone submission) ---
class UnidirectionalGRU(nn.Module):
    """Leakage-free unidirectional GRU for exp_06."""
    
    def __init__(self, input_dim=65, hidden_dim=256, num_layers=2, 
                 output_dim=2, dropout=0.3):
        super(UnidirectionalGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: No future peeking
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        predictions = self.fc(out)
        return predictions, hidden


class PredictionModel:
    """
    Exp 06: Leakage-Free Feature Engineering + Unidirectional GRU.
    
    Key principles:
    - Stateful GRU inference (matches training exactly)
    - Leakage-free feature computation
    - Efficient NumPy operations
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Determine model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            # Fallback to project root
            model_path = os.path.join(base_dir, "..", "..", "best_model.pth")
        
        self.device = torch.device("cpu")  # Inference on CPU
        
        # Initialize model
        self.input_dim = 65  # Will be verified during feature computation
        self.model = UnidirectionalGRU(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_layers=2,
            output_dim=2,
            dropout=0.3
        )
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # State for feature computation
        self.history = deque(maxlen=10)  # Max window is 10
        self.prev_vals = None  # For diffs
    
    def _compute_features(self, state):
        """
        Compute features from raw state (32 features).
        Must match features.py logic exactly.
        """
        epsilon = 1e-8
        s = state  # (32,)
        
        # Extract raw features
        p0, p1, p5, p6, p7 = s[0], s[1], s[5], s[6], s[7]
        v_bids = s[12:18]
        v_asks = s[18:24]
        
        # Base features
        mid0 = (p0 + p6) * 0.5
        mid1 = (p1 + p7) * 0.5
        spread0 = p6 - p0
        spread1 = p7 - p1
        
        # Volume imbalances
        vol_imbals = (v_bids - v_asks) / (v_bids + v_asks + epsilon)
        
        # Total volumes
        total_bid = np.sum(v_bids)
        total_ask = np.sum(v_asks)
        vol_imbal_total = (total_bid - total_ask) / (total_bid + total_ask + epsilon)
        
        # Ratios
        rel_spread = spread0 / (mid0 + epsilon)
        vol_ratio = total_bid / (total_ask + epsilon)
        price_depth = (p5 - p0) / (p0 + epsilon)
        
        # Update history for rolling features
        curr_targets = np.array([mid0, spread0, vol_imbal_total], dtype=np.float32)
        self.history.append(curr_targets)
        
        # Rolling features
        hist_arr = np.array(self.history)  # (N, 3)
        roll_feats = []
        
        for i in range(3):  # mid0, spread0, vol_imbal_total
            col_data = hist_arr[:, i]
            
            for w in [3, 10]:
                # Take last w elements
                window_data = col_data[-w:] if len(col_data) >= w else col_data
                
                # Mean
                mean = np.mean(window_data)
                
                # Std
                std = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
                
                roll_feats.extend([mean, std])
        
        # Diffs
        curr_diff_vals = np.array([p0, p6, mid0, spread0, vol_imbal_total], dtype=np.float32)
        if self.prev_vals is None:
            diffs = np.zeros(5, dtype=np.float32)
        else:
            diffs = curr_diff_vals - self.prev_vals
        self.prev_vals = curr_diff_vals
        
        # Concatenate all features
        # Order: [Raw 32] + [Base 15] + [Rolling 12] + [Diffs 5] = 64
        # Wait, let me recount:
        # Base: mid0, mid1, spread0, spread1 (4)
        #       vol_imbal_0..5 (6)
        #       total_bid, total_ask, vol_imbal_total (3)
        #       rel_spread, vol_ratio, price_depth (3)
        # Total base: 4 + 6 + 3 + 3 = 16
        # Rolling: 3 targets × 2 windows × 2 stats = 12
        # Diffs: 5
        # Total: 32 + 16 + 12 + 5 = 65 ✓
        
        features = np.concatenate([
            s,  # 32 raw features
            [mid0, mid1, spread0, spread1],  # 4
            vol_imbals,  # 6
            [total_bid, total_ask, vol_imbal_total],  # 3
            [rel_spread, vol_ratio, price_depth],  # 3
            np.array(roll_feats, dtype=np.float32),  # 12
            diffs  # 5
        ]).astype(np.float32)
        
        return features
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Predict for a single data point.
        
        Args:
            data_point: DataPoint with seq_ix, step_in_seq, need_prediction, state
        
        Returns:
            Prediction array of shape (2,) if need_prediction, else None
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
            self.history.clear()
            self.prev_vals = None
        
        # Compute features
        features = self._compute_features(data_point.state)
        
        # Prepare input tensor (1, 1, 65)
        x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Inference
        with torch.no_grad():
            output, self.hidden = self.model(x, self.hidden)
        
        if not data_point.need_prediction:
            return None
        
        # Return prediction (2,)
        prediction = output.squeeze().cpu().numpy()
        return prediction


if __name__ == "__main__":
    # Local testing
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing exp_06 solution...")
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("\nRunning validation...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Validation file not found at {test_file}")
        print("Please run from solutions/exp_06/ directory or ensure datasets are available")
