"""
Solution for exp_10: DeepLOB-style CNN-GRU Architecture

Inference wrapper for submission.
Uses raw 32 features (like exp_01) with CNN-GRU architecture.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add paths for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    try:
        from utils import DataPoint, ScorerStepByStep
    except ImportError:
        DataPoint = None
        ScorerStepByStep = None


# ============================================================================
# Model Architecture (must match src/model.py exactly)
# ============================================================================

class InceptionModule(nn.Module):
    """
    Inception-style module with multiple kernel sizes.
    """
    
    def __init__(self, in_channels, out_channels_per_branch, kernel_sizes=[1, 3, 5], dropout=0.2):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_channels_per_branch),
                nn.LeakyReLU(0.1),
            )
            self.branches.append(branch)
        
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels_per_branch * len(kernel_sizes)
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.dropout(out)
        return out


class DeepLOBGRU(nn.Module):
    """
    DeepLOB-style architecture: Inception CNN → GRU → Dual Heads
    Must match the trained model architecture exactly.
    """
    
    def __init__(
        self,
        input_dim=32,
        inception1_filters=16,
        inception2_filters=16,
        kernel_sizes=[1, 3, 5],
        gru_hidden=128,
        gru_layers=2,
        head_hidden=64,
        dropout_cnn=0.2,
        dropout_gru=0.4,
        dropout_t0=0.3,
        dropout_t1=0.5,
    ):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Inception Module 1
        self.inception1 = InceptionModule(
            in_channels=input_dim,
            out_channels_per_branch=inception1_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout_cnn
        )
        inception1_out = inception1_filters * len(kernel_sizes)  # 48
        
        # Inception Module 2
        self.inception2 = InceptionModule(
            in_channels=inception1_out,
            out_channels_per_branch=inception2_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout_cnn
        )
        inception2_out = inception2_filters * len(kernel_sizes)  # 48
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=inception2_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_gru if gru_layers > 1 else 0.0
        )
        
        # Layer norm after GRU
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        # t0 head
        self.t0_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t0),
            nn.Linear(head_hidden, 1)
        )
        
        # t1 head
        self.t1_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t1),
            nn.Linear(head_hidden, 1)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: GRU hidden state from previous step (optional)
        
        Returns:
            pred_t0: Predictions for t0 (batch, seq_len, 1)
            pred_t1: Predictions for t1 (batch, seq_len, 1)
            hidden: Updated GRU hidden state
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Normalize input
        x = self.input_norm(x)
        
        # Transpose for Conv1d: (batch, features, seq)
        x = x.transpose(1, 2)
        
        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        
        # Transpose back for GRU: (batch, seq, features)
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, hidden = self.gru(x, hidden)
        
        # Layer norm
        gru_out = self.gru_norm(gru_out)
        
        # Dual heads
        pred_t0 = self.t0_head(gru_out)
        pred_t1 = self.t1_head(gru_out)
        
        return pred_t0, pred_t1, hidden


# ============================================================================
# PredictionModel Class (required interface for submission)
# ============================================================================

class PredictionModel:
    """
    DeepLOB-style CNN-GRU Solution (exp_10).
    Stateful inference - maintains GRU hidden state across calls.
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Determine paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "..", "..", "best_model.pth")

        self.device = torch.device("cpu")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get model config from checkpoint
            model_config = checkpoint.get('model_config', {})
            
            # Create model with config (set dropout to 0 for inference)
            self.model = DeepLOBGRU(
                input_dim=model_config.get('input_dim', 32),
                inception1_filters=model_config.get('inception1_filters', 16),
                inception2_filters=model_config.get('inception2_filters', 16),
                kernel_sizes=model_config.get('kernel_sizes', [1, 3, 5]),
                gru_hidden=model_config.get('gru_hidden', 128),
                gru_layers=model_config.get('gru_layers', 2),
                head_hidden=model_config.get('head_hidden', 64),
                dropout_cnn=0.0,  # No dropout at inference
                dropout_gru=0.0,
                dropout_t0=0.0,
                dropout_t1=0.0,
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load normalization parameters
            if 'normalization' in checkpoint:
                norm = checkpoint['normalization']
                # Keys are 'mean' and 'std' (not 'means' and 'stds')
                self.means = torch.tensor(norm['mean'], dtype=torch.float32, device=self.device).squeeze()
                self.stds = torch.tensor(norm['std'], dtype=torch.float32, device=self.device).squeeze()
            else:
                self.means = torch.zeros(32, device=self.device)
                self.stds = torch.ones(32, device=self.device)
            
            print(f"Loaded DeepLOB-GRU model from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.means = torch.zeros(32, device=self.device)
            self.stds = torch.ones(32, device=self.device)

    def predict(self, data_point) -> np.ndarray:
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

        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None

        # Prepare input: data_point.state is a numpy array of 32 features
        x = torch.tensor(data_point.state, dtype=torch.float32, device=self.device)
        
        # Normalize
        x = (x - self.means) / (self.stds + 1e-8)
        
        # Add batch and sequence dimensions: (1, 1, 32)
        x = x.unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
        
        if not data_point.need_prediction:
            return None
        
        # Extract predictions
        t0 = pred_t0[0, 0, 0].item()
        t1 = pred_t1[0, 0, 0].item()
        
        return np.array([t0, t1])


if __name__ == "__main__":
    # Local testing
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file) and ScorerStepByStep is not None:
        print("Testing exp_10 solution...")
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("\nRunning validation...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in scorer.targets:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Validation file not found at {test_file} or ScorerStepByStep not available")
