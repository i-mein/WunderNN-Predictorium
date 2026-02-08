import numpy as np
import pandas as pd
import torch
import sys
import os

# Add src to path just in case, though usually not needed if installed as package
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.features import generate_features, get_raw_feature_columns, get_engineered_feature_columns

class Model:
    """
    Stateful inference model for exp_07 (Dual-Head Dual-Stream).
    
    Maintain:
    1. Feature Generation State (rolling windows, diffs)
    2. Raw Feature Buffer (for Causal CNN)
    3. RNN Hidden States (Eng Stream, t0 Head, t1 Head)
    """
    
    def __init__(self):
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.output_dir, "best_model.pth")
        self.device = torch.device("cpu")  # Inference on CPU usually
        
        # Load Model
        self._load_model()
        
        # Feature Generation State
        self.history_df = None
        self.max_history = 20  # Window 10 + safety
        
        # CNN Buffer (Receptive field ~13 steps)
        self.raw_buffer_size = 20 # Safe margin
        self.raw_buffer = [] # List of raw feature vectors
        
        # RNN Hidden States
        self.eng_hidden = None
        self.t0_hidden = None
        self.t1_hidden = None
        
        # Feature Columns
        print("Initializing feature columns...")
        dummy_df = pd.DataFrame([{f'p{i}': 1.0 for i in range(12)} | 
                                 {f'v{i}': 100 for i in range(12)} | 
                                 {f'dp{i}': 0.0 for i in range(4)} | 
                                 {f'dv{i}': 0.0 for i in range(4)} |
                                 {'seq_ix': 0, 'step_in_seq': 0}])
        dummy_feat = generate_features(dummy_df)
        self.raw_cols = get_raw_feature_columns()
        self.eng_cols = get_engineered_feature_columns(dummy_feat)
        
        print(f"Model ready. Raw: {len(self.raw_cols)}, Eng: {len(self.eng_cols)}")

    def _load_model(self):
        try:
            # Import here to avoid issues if src not in path
            from src.config import Config as CFG
            from src.model import DualHeadDualStream
            
            # Initialize model architecture
            # We need to know feature dims. We assume standard from training.
            # Raw: 32, Eng: 46 (as measured in tests)
            # Safe way: use known defaults or re-measure.
            # Since we can't measure without data, we use the values from config if available or defaults
            raw_dim = 32
            eng_dim = 46 # Hardcoded based on our feature generation testing
            
            print(f"Loading model from {self.model_path}")
            
            self.model = DualHeadDualStream(
                raw_dim=raw_dim,
                eng_dim=eng_dim,
                cnn_channels=CFG.CNN_CHANNELS,
                cnn_kernels=CFG.CNN_KERNELS,
                gru_hidden=CFG.GRU_HIDDEN,
                gru_layers=CFG.GRU_LAYERS,
                fusion_heads=CFG.FUSION_HEADS,
                head_hidden=CFG.HEAD_HIDDEN,
                head_attention_heads=CFG.HEAD_ATTENTION_HEADS,
                dropout_t0=0.0, # No dropout at inference
                dropout_t1=0.0
            )
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, test_df):
        """
        Make prediction for a single timestep.
        test_df contains raw LOB data for current step.
        """
        # 1. Update History for Feature Engineering
        if self.history_df is None:
            self.history_df = test_df
        else:
            self.history_df = pd.concat([self.history_df, test_df], axis=0)
        
        # Keep only necessary history
        if len(self.history_df) > self.max_history:
            self.history_df = self.history_df.iloc[-self.max_history:]
            
        # 2. Generate Features
        # "min_periods=1" in generate_features handles short history
        # sort_values is handled in generate_features, but for inference we just have sequential data
        # We assume test_df is a single row or batch of 1 row per sequence? 
        # The competition usually provides 1 row.
        
        feat_df = generate_features(self.history_df)
        
        # Extract features for CURRENT step (last row)
        current_feat = feat_df.iloc[-1:] 
        
        # 3. Prepare Tensors
        raw_vals = current_feat[self.raw_cols].values.astype(np.float32) # (1, 32)
        eng_vals = current_feat[self.eng_cols].values.astype(np.float32) # (1, 46)
        
        # 4. Update CNN Buffer
        self.raw_buffer.append(raw_vals)
        if len(self.raw_buffer) > self.raw_buffer_size:
            self.raw_buffer.pop(0)
            
        # Stack buffer for CNN
        # Shape: (1, seq_len, 32) where seq_len <= 13
        raw_seq = np.concatenate(self.raw_buffer, axis=0)
        raw_tensor = torch.from_numpy(raw_seq).unsqueeze(0).to(self.device) # (1, seq, 32)
        
        # Eng tensor: (1, 1, 46) - just current step for GRU
        eng_tensor = torch.from_numpy(eng_vals).unsqueeze(0).to(self.device)
        
        # 5. Inference
        with torch.no_grad():
            # Run CNN on buffer
            # Output: (1, seq, 128)
            # We take the last timestep: index -1
            raw_out_seq = self.model.raw_stream(raw_tensor)
            raw_out = raw_out_seq[:, -1:, :] # (1, 1, 128)
            
            # Run GRU using hidden state
            # Eng stream
            eng_out, self.eng_hidden = self.model.eng_stream(eng_tensor, self.eng_hidden)
            
            # Fusion
            # Concatenate: (1, 1, 384)
            fused = torch.cat([raw_out, eng_out], dim=-1)
            
            # Attention fusion (self-attention on length 1 is just projection+residual)
            fused = self.model.fusion(fused)
            
            # Task Heads
            pred_t0, self.t0_hidden = self.model.t0_head(fused, self.t0_hidden)
            pred_t1, self.t1_hidden = self.model.t1_head(fused, self.t1_hidden)
            
            # Extract scalars
            t0_val = pred_t0.item()
            t1_val = pred_t1.item()
            
        return t0_val, t1_val

def solve(test_df: pd.DataFrame, model_state=None):
    """
    Kaggle competition entry point.
    """
    if model_state is None:
        model_state = Model()
    
    t0, t1 = model_state.predict(test_df)
    return t0, t1, model_state
