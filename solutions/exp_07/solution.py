import numpy as np
import pandas as pd
import torch
import sys
import os
from collections import deque

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.features import generate_features, get_raw_feature_columns, get_engineered_feature_columns
    from src.model import DualHeadDualStream
    from src.config import Config as CFG
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    # Fallback for submission environment
    from features import generate_features, get_raw_feature_columns, get_engineered_feature_columns
    from model import DualHeadDualStream
    from config import Config as CFG
    from utils import DataPoint, ScorerStepByStep

class PredictionModel:
    """
    Exp 07: Dual-Head Dual-Stream Model.
    """
    
    def __init__(self, model_path=""):
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            model_path = os.path.join(self.output_dir, "best_model.pth")
            
        if not os.path.exists(model_path):
            # Try finding it relative to project root if running from elsewhere
            model_path = os.path.join(self.output_dir, "..", "..", "best_model.pth")
            
        self.device = torch.device("cpu")
        
        # Load Model
        self._load_model(model_path)
        
        # State
        self.current_seq_ix = None
        self.history_df = None
        self.max_history = 20  # Window safety margin
        
        # Feature Generation State
        # We need to reconstruct DataFrame columns from raw state array (32 values)
        # Raw features: p0..p11 (12), v0..v11 (12), dp0..dp3 (4), dv0..dv3 (4) = 32
        self.raw_col_names = []
        for i in range(12): self.raw_col_names.append(f'p{i}')
        for i in range(12): self.raw_col_names.append(f'v{i}')
        for i in range(4): self.raw_col_names.append(f'dp{i}')
        for i in range(4): self.raw_col_names.append(f'dv{i}')
        
        # CNN Buffer
        self.raw_buffer = [] 
        self.raw_buffer_size = 20
        
        # RNN Hidden States
        self.eng_hidden = None
        self.t0_hidden = None
        self.t1_hidden = None
        
        # Feature Columns (init once)
        self._init_feature_cols()

    def _init_feature_cols(self):
        print("Initializing feature columns...")
        dummy_data = {c: [0.0] for c in self.raw_col_names}
        dummy_data['seq_ix'] = [0]
        dummy_data['step_in_seq'] = [0]
        dummy_df = pd.DataFrame(dummy_data)
        
        try:
            dummy_feat = generate_features(dummy_df)
            self.eng_feat_cols = get_engineered_feature_columns(dummy_feat)
            self.raw_feat_cols = get_raw_feature_columns()
            print(f"Model ready. Raw: {len(self.raw_feat_cols)}, Eng: {len(self.eng_feat_cols)}")
        except Exception as e:
            print(f"Error initializing feature columns: {e}")
            # Fallback based on known counts if generation fails on empty
            self.eng_feat_cols = [] 

    def _load_model(self, model_path):
        try:
            # Initialize model architecture matching training config
            # We assume dimensions match what was trained
            # Raw: 32, Eng: 46 (dynamic)
            # Safe default for Eng check
            eng_dim = 46 
            
            print(f"Loading model from {model_path}")
            
            self.model = DualHeadDualStream(
                raw_dim=32,
                eng_dim=eng_dim,
                cnn_channels=CFG.CNN_CHANNELS,
                cnn_kernels=CFG.CNN_KERNELS,
                gru_hidden=CFG.GRU_HIDDEN,
                gru_layers=CFG.GRU_LAYERS,
                fusion_heads=CFG.FUSION_HEADS,
                head_hidden=CFG.HEAD_HIDDEN,
                head_attention_heads=CFG.HEAD_ATTENTION_HEADS,
                dropout_t0=0.0,
                dropout_t1=0.0
            )
            
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Make prediction for a single data point.
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history_df = None
            self.raw_buffer = []
            self.eng_hidden = None
            self.t0_hidden = None
            self.t1_hidden = None
            
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None

        # 1. Convert state to DataFrame row
        # state is (32,) numpy array
        row_dict = {col: [val] for col, val in zip(self.raw_col_names, data_point.state)}
        row_dict['seq_ix'] = [data_point.seq_ix]
        row_dict['step_in_seq'] = [data_point.step_in_seq]
        
        current_df = pd.DataFrame(row_dict)
        
        # 2. Update History
        if self.history_df is None:
            self.history_df = current_df
        else:
            self.history_df = pd.concat([self.history_df, current_df], axis=0)
        
        # Keep window
        if len(self.history_df) > self.max_history:
            self.history_df = self.history_df.iloc[-self.max_history:]
            
        # 3. Generate Features
        # This recomputes features for the whole history window every step
        # It's less efficient than incremental numpy but robust and uses exact same logic as training
        try:
            feat_df = generate_features(self.history_df)
            current_feat = feat_df.iloc[-1:]
            
            raw_vals = current_feat[self.raw_feat_cols].values.astype(np.float32) # (1, 32)
            eng_vals = current_feat[self.eng_feat_cols].values.astype(np.float32) # (1, 46)
        except Exception as e:
            print(f"Error generating features: {e}")
            return np.zeros(2) if data_point.need_prediction else None
            
        # 4. Update CNN Buffer
        self.raw_buffer.append(raw_vals)
        if len(self.raw_buffer) > self.raw_buffer_size:
            self.raw_buffer.pop(0)
            
        # Stack buffer for CNN
        raw_seq = np.concatenate(self.raw_buffer, axis=0)
        raw_tensor = torch.from_numpy(raw_seq).unsqueeze(0).to(self.device) # (1, seq, 32)
        eng_tensor = torch.from_numpy(eng_vals).unsqueeze(0).to(self.device) # (1, 1, 46)
        
        # 5. Inference
        with torch.no_grad():
            # Run CNN
            raw_out_seq = self.model.raw_stream(raw_tensor)
            raw_out = raw_out_seq[:, -1:, :] # Take last step
            
            # Run GRU
            eng_out, self.eng_hidden = self.model.eng_stream(eng_tensor, self.eng_hidden)
            
            # Fuse
            fused = torch.cat([raw_out, eng_out], dim=-1)
            fused = self.model.fusion(fused)
            
            # Heads
            pred_t0, self.t0_hidden = self.model.t0_head(fused, self.t0_hidden)
            pred_t1, self.t1_hidden = self.model.t1_head(fused, self.t1_hidden)
            
            # Output
            t0_val = pred_t0.item()
            t1_val = pred_t1.item()
            
        if not data_point.need_prediction:
            return None
            
        return np.array([t0_val, t1_val])

if __name__ == "__main__":
    # Local testing
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing exp_07 solution...")
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
