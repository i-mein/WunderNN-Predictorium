"""
exp_23b: Multi-Stream TCN + GRU with Attention
Solution for LOB price movement prediction.

Architecture:
- 4 separate TCN streams for: prices, volumes, deltas, depth features
- Cross-stream attention for fusion
- Temporal attention for causal self-attention
- GRU for sequential processing
- Separate heads for t0 and t1 prediction
"""

import os
import sys
import numpy as np
import torch

# Add src directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.model import MultiStreamTCNGRUModel
from src.features import compute_enhanced_features
from src.config import CONFIG, get_model_kwargs


class DataPoint:
    """Simplified DataPoint for type hints."""
    def __init__(self, state, seq_ix, step_in_seq, need_prediction):
        self.state = state
        self.seq_ix = seq_ix
        self.step_in_seq = step_in_seq
        self.need_prediction = need_prediction


class PredictionModel:
    """
    Multi-Stream TCN + GRU prediction model with causal normalization.
    
    Processes one timestep at a time, maintaining hidden state across
    the sequence and using online EMA normalization.
    """
    
    def __init__(self):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        # Load model
        self.model = MultiStreamTCNGRUModel(**get_model_kwargs(for_inference=True))
        
        # Load weights
        model_path = os.path.join(SCRIPT_DIR, "best_model.pth")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # State tracking
        self.current_seq_ix = None
        self.hidden = None
        
        # Online normalization state
        self.norm_alpha = CONFIG.get("norm_ema_alpha", 0.01)
        self.running_mean = None
        self.running_var = None
        
        # Feature dimension after enhancement
        self.feature_dim = 47
    
    def _reset_state(self):
        """Reset all stateful components for new sequence."""
        self.hidden = None
        self.running_mean = None
        self.running_var = None
    
    def _normalize_online(self, features):
        """
        Online (causal) normalization using EMA.
        Matches training normalization exactly.
        """
        if self.running_mean is None:
            self.running_mean = features.copy()
            self.running_var = np.ones(self.feature_dim, dtype=np.float32)
        
        # Normalize using current running statistics
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (features - self.running_mean) / std
        
        # Update running statistics
        delta = features - self.running_mean
        self.running_mean = self.running_mean + self.norm_alpha * delta
        self.running_var = (1 - self.norm_alpha) * self.running_var + self.norm_alpha * (delta ** 2)
        
        return normalized
    
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Generate predictions for a single timestep.
        
        Args:
            data_point: Contains state (32 raw features), seq_ix, step_in_seq, need_prediction
            
        Returns:
            np.ndarray of shape (2,) with [t0_pred, t1_pred] or None if need_prediction is False
        """
        # Check for new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self._reset_state()
        
        # Get raw state
        raw_state = np.array(data_point.state, dtype=np.float32)
        
        # Compute enhanced features (32 -> 47)
        enhanced = compute_enhanced_features(raw_state)
        
        # Apply causal normalization
        normalized = self._normalize_online(enhanced)
        
        # Prepare input tensor: (batch=1, seq_len=1, features)
        input_tensor = torch.tensor(
            normalized.reshape(1, 1, -1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward pass
        with torch.no_grad():
            output, self.hidden = self.model(input_tensor, self.hidden)
        
        # Return prediction only if needed
        if not data_point.need_prediction:
            return None
        
        # Extract prediction for this timestep
        pred = output[0, 0, :].cpu().numpy()  # Shape (2,)
        
        return pred


# ============================================
# Full Validation (for testing)
# ============================================
if __name__ == "__main__":
    import time
    
    # Add project root to path for utils
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    sys.path.insert(0, PROJECT_ROOT)
    
    from src.utils import ScorerStepByStep
    
    print("=" * 60)
    print("exp_23b: Full Validation using ScorerStepByStep")
    print("=" * 60)
    
    valid_path = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    
    print(f"Loading validation data from {valid_path}...")
    scorer = ScorerStepByStep(valid_path)
    
    print("Running inference...")
    start_time = time.time()
    
    model = PredictionModel()
    result = scorer.score(model)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Full-Val Results:")
    print(f"  t0_corr: {result['t0_corr']:.4f}")
    print(f"  t1_corr: {result['t1_corr']:.4f}")
    print(f"  Weighted Score: {result['weighted_score']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")
