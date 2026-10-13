"""
Solution for exp_12: Specialized t1 Prediction

Key insight: t0 (~0.37 Pearson) is easier than t1 (~0.14 Pearson).
Asymmetric heads: simple t0 head, deeper t1 head.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    from utils import DataPoint, ScorerStepByStep


# --- MODEL DEFINITION (Copied for standalone submission) ---

class AsymmetricDualHeadGRU(nn.Module):
    """
    GRU with asymmetric dual heads for t0/t1 prediction.
    
    Architecture:
    - Shared GRU backbone
    - Simple t0 head (1 hidden layer)
    - Deep t1 head (2 hidden layers, more capacity)
    """
    
    def __init__(
        self,
        input_dim=32,
        gru_hidden=256,
        gru_layers=2,
        gru_dropout=0.3,
        t0_hidden=64,
        t0_dropout=0.2,
        t1_hidden1=128,
        t1_hidden2=64,
        t1_dropout=0.4,
    ):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Shared GRU backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout if gru_layers > 1 else 0.0
        )
        
        # Layer norm after GRU
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        # Simple t0 head (1 hidden layer)
        self.t0_head = nn.Sequential(
            nn.Linear(gru_hidden, t0_hidden),
            nn.GELU(),
            nn.Dropout(t0_dropout),
            nn.Linear(t0_hidden, 1)
        )
        
        # Deep t1 head (2 hidden layers for more capacity)
        self.t1_head = nn.Sequential(
            nn.Linear(gru_hidden, t1_hidden1),
            nn.GELU(),
            nn.Dropout(t1_dropout),
            nn.Linear(t1_hidden1, t1_hidden2),
            nn.GELU(),
            nn.Dropout(t1_dropout),
            nn.Linear(t1_hidden2, 1)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            hidden: GRU hidden state
        
        Returns:
            pred_t0: (batch, seq_len, 1)
            pred_t1: (batch, seq_len, 1)
            hidden: Updated hidden state
        """
        # Input normalization
        x = self.input_norm(x)
        
        # GRU
        gru_out, hidden = self.gru(x, hidden)
        
        # Layer norm
        gru_out = self.gru_norm(gru_out)
        
        # Dual heads
        pred_t0 = self.t0_head(gru_out)
        pred_t1 = self.t1_head(gru_out)
        
        return pred_t0, pred_t1, hidden


class PredictionModel:
    """
    Exp 12: Asymmetric Dual Head GRU for t1-focused prediction.
    
    Key principles:
    - Stateful GRU inference (matches training exactly)
    - Simple t0 head (task is easier)
    - Deeper t1 head (task is harder)
    - Raw 32 features only
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Determine model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "..", "..", "best_model.pth")
        
        self.device = torch.device("cpu")
        
        # Initialize model with config from training
        self.model = AsymmetricDualHeadGRU(
            input_dim=32,
            gru_hidden=256,
            gru_layers=2,
            gru_dropout=0.3,
            t0_hidden=64,
            t0_dropout=0.2,
            t1_hidden1=128,
            t1_hidden2=64,
            t1_dropout=0.4,
        )
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model state dict: {model_path}")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
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
        
        # Prepare input tensor (1, 1, 32)
        x = torch.from_numpy(data_point.state.astype(np.float32))
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Inference
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
        
        if not data_point.need_prediction:
            return None
        
        # Return prediction (2,)
        t0 = pred_t0.squeeze().cpu().item()
        t1 = pred_t1.squeeze().cpu().item()
        
        return np.array([t0, t1], dtype=np.float32)


if __name__ == "__main__":
    # Local testing
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing exp_12 solution...")
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
        print("Please run from solutions/exp_12/ directory or ensure datasets are available")
