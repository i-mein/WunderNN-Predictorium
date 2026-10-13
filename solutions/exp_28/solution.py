"""
exp_28: Advanced Transformer Solution for LOB Prediction

Architecture from aicheye/wundernn adapted for:
- 2 regression targets (t0, t1) 
- Stateful inference with sequence history
- Causal (online) normalization

Inference pattern:
1. Reset state on new sequence
2. Accumulate history for context window
3. Apply online normalization (matches training)
4. Run transformer prediction
"""

import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

from src.utils import DataPoint
from src.config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from src.model import AdvancedTransformerPredictor


class PredictionModel:
    """
    Advanced Transformer prediction model for LOB.
    
    Uses a context window of past timesteps for transformer attention.
    Applies causal (online) normalization matching training.
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.context_len = MODEL_CONFIG["context_len"]
        
        # History buffer for context window
        self.sequence_history = []
        
        # Online normalization state (EMA)
        self.running_mean = None
        self.running_var = None
        self.norm_alpha = TRAIN_CONFIG.get("norm_ema_alpha", 0.01)
        
        # Device setup
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        # Load model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        # Try alternative paths
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "best_model_corr.pth")
        
        self.model = AdvancedTransformerPredictor(**get_model_kwargs(for_inference=True))
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _normalize_features(self, state: np.ndarray) -> np.ndarray:
        """
        Apply causal (online) normalization using EMA.
        
        This EXACTLY matches training normalization to prevent leakage.
        Only uses past information for normalization.
        """
        # Clean input
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.norm_alpha
        
        if self.running_mean is None:
            # Initialize with first observation
            self.running_mean = state.copy()
            self.running_var = np.ones_like(state, dtype=np.float32)
            
            # First timestep: normalized to 0
            normalized = np.zeros_like(state)
        else:
            # Normalize using PREVIOUS statistics (causal)
            std = np.sqrt(self.running_var + 1e-8)
            normalized = (state - self.running_mean) / std
        
        # Update running statistics
        delta = state - self.running_mean
        self.running_mean = self.running_mean + alpha * delta
        self.running_var = (1 - alpha) * self.running_var + alpha * (delta ** 2)
        
        # Clean output
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Generate prediction for current timestep.
        
        Args:
            data_point: Current market state with seq_ix, step_in_seq, need_prediction, state
            
        Returns:
            np.ndarray of shape (2,) with predictions for t0, t1
            None if prediction not needed
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
            self.running_mean = None
            self.running_var = None
        
        # Normalize current state (causally)
        normalized_state = self._normalize_features(data_point.state)
        
        # Add to history
        self.sequence_history.append(normalized_state.copy())
        
        # Return None if prediction not needed
        if not data_point.need_prediction:
            return None
        
        # Fallback if model not loaded
        if self.model is None:
            return np.zeros(2)
        
        # Prepare context window
        context = np.array(self.sequence_history[-self.context_len:])
        
        # Pad if necessary
        if context.shape[0] < self.context_len:
            pad_len = self.context_len - context.shape[0]
            context = np.pad(
                context,
                ((pad_len, 0), (0, 0)),
                mode='edge'
            )
        
        # Convert to tensor: [1, context_len, input_dim]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        context_tensor = context_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(context_tensor, return_attention=False)
            prediction = output["prediction"]
        
        # Return as numpy array [t0, t1]
        return prediction.squeeze().cpu().numpy()


# ============================================
# Local Testing
# ============================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
    from src.utils import ScorerStepByStep
    
    test_file = os.path.join(SCRIPT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing Advanced Transformer Solution (exp_28)...")
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in ['t0', 't1']:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Valid parquet not found at {test_file}")
