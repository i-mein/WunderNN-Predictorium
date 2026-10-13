"""
exp_27: LOB-ConvNeXt - ConvNeXt-based architecture for LOB prediction
"""

import os
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific PyTorch warnings
# Add src to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from src.utils import DataPoint
from config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from model_convnext import LOBViTModel


class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden_states = []  # One hidden state per model
        self.feature_buffer = None
        self.window_size = MODEL_CONFIG["window_size"]
        
        self.running_mean = None
        self.running_var = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # self.device = torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        
        # Load snapshot models for ensemble
        # Priority: 1) averaged_model.pth, 2) best_model_ep_*.pth, 3) best_model.pth
        self.models = []
        
        # Check for averaged model first (created after full validation)
        averaged_path = os.path.join(base_dir, "averaged_model.pth")
        if os.path.exists(averaged_path):
            snapshot_files = ["averaged_model.pth"]
        else:
            # Look for epoch-specific snapshots
            import glob
            ep_snapshots = glob.glob(os.path.join(base_dir, "best_model_ep_*.pth"))
            if ep_snapshots:
                snapshot_files = [os.path.basename(f) for f in sorted(ep_snapshots)]
            else:
                # Fallback to best_model.pth
                snapshot_files = ["best_model.pth"]
        
        for snapshot_file in snapshot_files:
            model_path = os.path.join(base_dir, snapshot_file)
            if not os.path.exists(model_path):
                continue
            
            model = LOBViTModel(**get_model_kwargs(for_inference=True))
            
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"Loaded snapshot: {snapshot_file}")
            except Exception as e:
                print(f"Error loading {snapshot_file}: {e}")
        
        print(f"Loaded {len(self.models)} models for ensemble")
        self.hidden_states = [None] * len(self.models)
        
        # Get normalization alpha from config
        self.norm_alpha = TRAIN_CONFIG.get("norm_ema_alpha", 0.01)
    
    def _normalize_features(self, state):
        """
        Causal (online) normalization using EMA.
        
        This matches training behavior exactly - only uses past information
        at each timestep. Prevents normalization leakage.
        """
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.norm_alpha
        
        if self.running_mean is None:
            # Initialize with first observation - matches training exactly
            self.running_mean = state.copy()
            self.running_var = np.ones_like(state, dtype=np.float32)
            
            # First timestep: normalized to 0
            normalized = np.zeros_like(state)
            
            # Still do the update (delta=0, so running_mean unchanged, running_var becomes 1-alpha)
            delta = state - self.running_mean  # = 0
            self.running_mean = self.running_mean + alpha * delta  # unchanged
            self.running_var = (1 - alpha) * self.running_var + alpha * (delta ** 2)
            
            return normalized
        
        # Normalize using current running statistics (before update)
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (state - self.running_mean) / std
        
        # Update running statistics for next timestep (EMA)
        delta = state - self.running_mean
        self.running_mean = self.running_mean + alpha * delta
        self.running_var = (1 - alpha) * self.running_var + alpha * (delta ** 2)
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_states = [None] * len(self.models)
            self.feature_buffer = None
            self.running_mean = None
            self.running_var = None
        
        if not self.models:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Normalize and add to buffer
        normalized_state = self._normalize_features(data_point.state)
        
        if self.feature_buffer is None:
            self.feature_buffer = normalized_state.reshape(1, 32)
        else:
            self.feature_buffer = np.vstack([self.feature_buffer, normalized_state.reshape(1, 32)])
            if self.feature_buffer.shape[0] > self.window_size:
                self.feature_buffer = self.feature_buffer[-self.window_size:]
        
        # Create window tensor: (1, window_size, 32)
        window_len = self.feature_buffer.shape[0]
        if window_len < self.window_size:
            padding = np.zeros((self.window_size - window_len, 32))
            window = np.vstack([padding, self.feature_buffer])
        else:
            window = self.feature_buffer
        
        # Tensor shape: (1, window_size, 32)
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get predictions from all models and average
        all_predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                # forward() handles reshaping internally via LOBReshaper
                # Input: (B, W, 32), Output: (B, 2)
                output, self.hidden_states[i] = model(window_tensor, self.hidden_states[i])
                all_predictions.append(output.detach().cpu().numpy().flatten())
        
        if not data_point.need_prediction:
            return None
        
        # Average predictions from all models
        ensemble_prediction = np.mean(all_predictions, axis=0)
        return np.clip(ensemble_prediction, -6.0, 6.0)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from src.utils import ScorerStepByStep
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--average-weights", nargs="+", help="Checkpoint files to average weights from")
    parser.add_argument("--validate-all", action="store_true", help="Validate all epoch checkpoints")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.average_weights:
        # Average weights from specified checkpoints
        print(f"Averaging weights from {len(args.average_weights)} checkpoints...")
        
        state_dicts = []
        for ckpt_file in args.average_weights:
            ckpt_path = os.path.join(base_dir, ckpt_file)
            if os.path.exists(ckpt_path):
                state_dicts.append(torch.load(ckpt_path, map_location="cpu", weights_only=True))
                print(f"  Loaded: {ckpt_file}")
            else:
                print(f"  Not found: {ckpt_file}")
        
        if state_dicts:
            # Average the weights
            averaged_state_dict = {}
            for key in state_dicts[0].keys():
                averaged_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
            
            # Save averaged model
            output_path = os.path.join(base_dir, "averaged_model.pth")
            torch.save(averaged_state_dict, output_path)
            print(f"\nAveraged model saved to: {output_path}")
    
    elif args.validate_all:
        # Validate all epoch checkpoints individually
        import glob
        ep_checkpoints = sorted(glob.glob(os.path.join(base_dir, "best_model_ep_*.pth")))
        
        if not ep_checkpoints:
            print("No epoch checkpoints found (best_model_ep_*.pth)")
        else:
            print(f"Validating {len(ep_checkpoints)} checkpoints...\n")
            results_list = []
            
            for ckpt_path in ep_checkpoints:
                ckpt_name = os.path.basename(ckpt_path)
                
                # Create single-model predictor
                class SingleModelPredictor(PredictionModel):
                    def __init__(self, checkpoint_path):
                        self.current_seq_ix = None
                        self.hidden_states = [None]
                        self.feature_buffer = None
                        self.window_size = MODEL_CONFIG["window_size"]
                        self.running_mean = None
                        self.running_var = None
                        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                        self.norm_alpha = TRAIN_CONFIG.get("norm_ema_alpha", 0.01)
                        print(f"Using device: {self.device}")

                        model = LOBViTModel(**get_model_kwargs(for_inference=True))
                        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                        model.load_state_dict(state_dict)
                        model.to(self.device)
                        model.eval()
                        self.models = [model]
                
                predictor = SingleModelPredictor(ckpt_path)
                scorer = ScorerStepByStep("datasets/valid.parquet")
                results = scorer.score(predictor)
                
                print(f"{ckpt_name}: t0={results['t0']:.4f}, t1={results['t1']:.4f}, weighted={results['weighted_pearson']:.4f}")
                results_list.append((ckpt_name, results))
            
            print("\n" + "="*60)
            print("Summary (sorted by weighted score):")
            results_list.sort(key=lambda x: x[1]['weighted_pearson'], reverse=True)
            for name, res in results_list:
                print(f"  {name}: {res['weighted_pearson']:.4f}")
    
    else:
        # Default: run full validation
        print("Testing exp_27 LOB-ConvNeXt...")
        model = PredictionModel()
        scorer = ScorerStepByStep("datasets/valid.parquet")
        results = scorer.score(model)
        
        print(f"\nResults:")
        print(results)
