import os
import sys
import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint, ScorerStepByStep

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: exp_01 is in solutions/exp_01, so utils is in ../../
sys.path.append(f"{CURRENT_DIR}/../..")

# --- MODEL DEFINITION ---
# Copied from src/model.py to ensure standalone submission works without src package structure issues
class GRUModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer GRU
        # batch_first=True expects input of shape (batch, seq, feature)
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        predictions = self.fc(out)
        return predictions, hidden

class PredictionModel:
    """
    PyTorch GRU Solution (Exp 01).
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        
        # Determine paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for model in the same directory
        model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
             # Try fallback to project root if running locally/during training
              model_path = os.path.join(base_dir, "..", "..", "best_model.pth")

        self.device = torch.device("cpu") # Inference on CPU usually
        
        self.model = GRUModel(input_dim=32, hidden_dim=256, num_layers=2, output_dim=2)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None 
            # Note: We don't keep history buffer anymore because GRU is stateful!
            # We feed one step at a time and keep 'hidden' state.

        # Prepare input
        # Input shape: (Batch=1, Seq=1, Features=32)
        state_tensor = torch.tensor(data_point.state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        if self.model is None:
             return np.zeros(2) if data_point.need_prediction else None

        with torch.no_grad():
            output, self.hidden = self.model(state_tensor, self.hidden)
        
        if not data_point.need_prediction:
            return None
            
        # Output shape: (1, 1, 2)
        prediction = output.squeeze().numpy()
        return prediction

if __name__ == "__main__":
    # Local testing
    # Assuming we are in solutions/exp_01/
    test_file = f"{CURRENT_DIR}/../../datasets/valid.parquet"
    
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Testing PyTorch GRU Solution (Exp 01)...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print(f"Valid parquet not found for testing at {test_file}")
