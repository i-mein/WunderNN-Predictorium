"""
exp_29: LOBViT 2D Transformer - Solution for inference

2D reshaping: Raw 32 features → (B, 12, T, 3)
- 12 LOB levels (6 bid + 6 ask)  
- T timesteps
- 3 channels (price, volume, temporal_delta)
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

from src.utils import DataPoint
from src.config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from src.model import LOBViT


def reshape_raw_to_2d_np(raw_features: np.ndarray) -> np.ndarray:
    """
    Reshape raw features to 2D format (numpy version for inference).
    
    Args:
        raw_features: (T, 32) raw LOB features
        
    Returns:
        (12, T, 3) reshaped features
    """
    T, F = raw_features.shape
    assert F == 32, f"Expected 32 features, got {F}"
    
    # Prices: (T, 12)
    prices = np.concatenate([
        raw_features[:, 0:6],   # Bid prices
        raw_features[:, 6:12],  # Ask prices
    ], axis=1)
    
    # Volumes: (T, 12)
    volumes = np.concatenate([
        raw_features[:, 12:18],  # Bid volumes
        raw_features[:, 18:24],  # Ask volumes
    ], axis=1)
    
    # Temporal delta (causal): delta[t] = price[t] - price[t-1]
    temporal_delta = np.zeros_like(prices)
    temporal_delta[1:] = prices[1:] - prices[:-1]
    
    # Stack: (T, 12, 3)
    x = np.stack([prices, volumes, temporal_delta], axis=2)
    
    # Permute to (12, T, 3)
    x = np.transpose(x, (1, 0, 2))
    
    return x


class PredictionModel:
    """LOBViT 2D Transformer for LOB prediction."""
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.context_len = MODEL_CONFIG["context_len"]
        
        # History buffer
        self.sequence_history = []
        
        # Online normalization state
        self.running_mean = None
        self.running_var = None
        self.norm_alpha = TRAIN_CONFIG.get("norm_ema_alpha", 0.01)
        
        # Device
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        # Load model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "best_model_corr.pth")
        
        self.model = LOBViT(**get_model_kwargs(for_inference=True))
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded LOBViT model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _normalize_features(self, state: np.ndarray) -> np.ndarray:
        """Causal EMA normalization."""
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.norm_alpha
        
        if self.running_mean is None:
            self.running_mean = state.copy()
            self.running_var = np.ones_like(state, dtype=np.float32)
            normalized = np.zeros_like(state)
        else:
            std = np.sqrt(self.running_var + 1e-8)
            normalized = (state - self.running_mean) / std
        
        delta = state - self.running_mean
        self.running_mean = self.running_mean + alpha * delta
        self.running_var = (1 - alpha) * self.running_var + alpha * (delta ** 2)
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
            self.running_mean = None
            self.running_var = None
        
        # Normalize and store
        normalized = self._normalize_features(data_point.state)
        self.sequence_history.append(normalized.copy())
        
        if not data_point.need_prediction:
            return None
        
        if self.model is None:
            return np.zeros(2)
        
        # Get context window
        context = np.array(self.sequence_history[-self.context_len:])
        
        # Pad if needed
        if context.shape[0] < self.context_len:
            pad_len = self.context_len - context.shape[0]
            context = np.pad(context, ((pad_len, 0), (0, 0)), mode='edge')
        
        # Reshape to 2D: (T, 32) → (12, T, 3)
        context_2d = reshape_raw_to_2d_np(context)
        
        # Convert to tensor: (1, 12, T, 3)
        context_tensor = torch.tensor(context_2d, dtype=torch.float32).unsqueeze(0)
        context_tensor = context_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(context_tensor)
        
        return prediction.squeeze().cpu().numpy()


# ============================================
# Local Testing
# ============================================
if __name__ == "__main__":
    # Import from project root utils (not local src/utils)
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
    sys.path.insert(0, PROJECT_ROOT)
    from src.utils import ScorerStepByStep
    
    test_file = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing LOBViT 2D Transformer (exp_29)...")
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in ['t0', 't1']:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Valid parquet not found at {test_file}")
