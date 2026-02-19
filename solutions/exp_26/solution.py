"""
exp_26: LOB Vision Transformer - Causal Normalization

Improvements over exp_25b:
1. Window size: 100 timesteps (full warmup period)
2. Causal normalization: Online EMA matching training exactly
3. Attention pooling: Learnable attention over level patches
4. Better patch sizes: 3x5 (4 level patches x 20 time patches)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DataPoint

# Add src to path for config import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs


class LOBReshaper(nn.Module):
    """Reshape raw 32 LOB features into 2D spatial format (H, W, C)."""
    
    def __init__(self, num_levels=12, num_channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = num_channels
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 32) raw LOB features
        Returns:
            reshaped: (batch, seq_len, H=12, C=3)
        """
        batch, seq_len, _ = x.shape
        out = torch.zeros(batch, seq_len, self.num_levels, self.num_channels,
                         device=x.device, dtype=x.dtype)
        
        # Channel 0: Prices (bid p0-p5 -> rows 0-5, ask p6-p11 -> rows 6-11)
        out[:, :, 0:6, 0] = x[:, :, 0:6]
        out[:, :, 6:12, 0] = x[:, :, 6:12]
        
        # Channel 1: Volumes (bid v0-v5 -> rows 0-5, ask v6-v11 -> rows 6-11)
        out[:, :, 0:6, 1] = x[:, :, 12:18]
        out[:, :, 6:12, 1] = x[:, :, 18:24]
        
        # Channel 2: Deltas (dp0-dp3 -> rows 0-3, dv0-dv3 -> rows 4-7)
        out[:, :, 0:4, 2] = x[:, :, 24:28]
        out[:, :, 4:8, 2] = x[:, :, 28:32]
        
        return out


class LOB2DPatchEmbedding(nn.Module):
    """2D Patch Embedding for LOB data."""
    
    def __init__(self, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.embed_dim = embed_dim
        
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        patch_dim = patch_size_h * patch_size_w * num_channels
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, H=12, W=window_size, C=3)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        batch, H, W, C = x.shape
        
        if W < self.window_size:
            pad_w = self.window_size - W
            x = F.pad(x, (0, 0, pad_w, 0))
            W = self.window_size
        elif W > self.window_size:
            x = x[:, :, -self.window_size:, :]
            W = self.window_size
        
        x = x.view(batch, self.num_patches_h, self.patch_size_h,
                   self.num_patches_w, self.patch_size_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, self.num_patches, -1)
        
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class LearnablePositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for patches.
    Includes both separate h/w embeddings and single position embedding for compatibility.
    """
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Separate embeddings for height and width (for checkpoint compatibility)
        self.pos_embed_h = nn.Embedding(num_patches_h, embed_dim // 2)
        self.pos_embed_w = nn.Embedding(num_patches_w, embed_dim // 2)
        
        # Single embedding for all positions (used in forward)
        self.pos_embed = nn.Embedding(num_patches_h * num_patches_w, embed_dim)
    
    def forward(self, x):
        batch, num_patches, _ = x.shape
        positions = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch, -1)
        return x + self.pos_embed(positions)


class CausalTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with causal masking."""
    
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CausalTransformerEncoder(nn.Module):
    """Stack of transformer encoder layers with causal masking."""
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.0,
                 num_patches_h=3, num_patches_w=5):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask_2d(self, num_patches, device):
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        
        mask = torch.zeros(num_patches, num_patches, device=device)
        for i in range(num_patches):
            i_w = i % num_w
            for j in range(num_patches):
                j_w = j % num_w
                if j_w > i_w:
                    mask[i, j] = float('-inf')
        return mask
    
    def forward(self, x):
        num_patches = x.shape[1]
        attn_mask = self.generate_causal_mask_2d(num_patches, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class TemporalPooling(nn.Module):
    """Pool patch embeddings along level dimension."""
    
    def __init__(self, num_patches_h, num_patches_w, embed_dim, pool_type='mean'):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            self.attention = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        batch, num_patches, embed_dim = x.shape
        x = x.view(batch, self.num_patches_h, self.num_patches_w, embed_dim)
        
        if self.pool_type == 'mean':
            pooled = x.mean(dim=1)
        elif self.pool_type == 'max':
            pooled = x.max(dim=1)[0]
        elif self.pool_type == 'attention':
            attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return pooled


class LOBViTModel(nn.Module):
    """LOB Vision Transformer for regression."""
    
    def __init__(self, input_dim=32, num_levels=12, num_channels=3, window_size=10,
                 patch_size_h=4, patch_size_w=2, embed_dim=64,
                 num_transformer_layers=2, num_heads=4, ff_dim=256,
                 gru_hidden=128, gru_layers=2, output_dim=2, dropout=0.0,
                 pool_type='mean'):
        super().__init__()
        
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        self.num_patches_h = num_levels // patch_size_h
        self.num_patches_w = window_size // patch_size_w
        
        self.reshaper = LOBReshaper(num_levels, num_channels)
        self.patch_embed = LOB2DPatchEmbedding(
            num_levels, num_channels, window_size, patch_size_h, patch_size_w, embed_dim
        )
        self.pos_encoding = LearnablePositionalEncoding2D(
            self.num_patches_h, self.num_patches_w, embed_dim
        )
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout,
            self.num_patches_h, self.num_patches_w
        )
        self.temporal_pool = TemporalPooling(
            self.num_patches_h, self.num_patches_w, embed_dim, pool_type
        )
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward_single_window(self, window_2d, hidden=None):
        """
        Process a single window for incremental inference.
        
        Args:
            window_2d: (batch, H=12, W=window_size, C=3)
            hidden: GRU hidden state
        Returns:
            predictions: (batch, 2)
            hidden: updated hidden state
        """
        # Patch embedding
        patches = self.patch_embed(window_2d)
        patches = self.pos_encoding(patches)
        
        # Transformer
        transformer_out = self.transformer(patches)
        
        # Pool to get per-timestep features, take last
        pooled = self.temporal_pool(transformer_out)
        pooled = pooled[:, -1:, :]  # (batch, 1, embed_dim)
        
        # GRU step
        gru_out, hidden = self.gru(pooled, hidden)
        
        # Output
        predictions = self.output_head(gru_out.squeeze(1))
        
        return predictions, hidden


class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden_states = []  # One hidden state per model
        self.feature_buffer = None
        self.window_size = MODEL_CONFIG["window_size"]
        
        self.running_mean = None
        self.running_var = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cpu")
        
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
            # This matches training where we iterate through t=0
            delta = state - self.running_mean  # = 0
            self.running_mean = self.running_mean + alpha * delta  # unchanged
            self.running_var = (1 - alpha) * self.running_var + alpha * (delta ** 2)  # = 1-alpha
            
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
        
        # Create window tensor
        window_len = self.feature_buffer.shape[0]
        if window_len < self.window_size:
            padding = np.zeros((self.window_size - window_len, 32))
            window = np.vstack([padding, self.feature_buffer])
        else:
            window = self.feature_buffer
        
        # Reshape to (1, seq_len, 32)
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions from all models and average
        all_predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                # Reshape to 2D: (1, H=12, W=window_size, C=3)
                window_2d = model.reshaper(window_tensor)
                window_2d = window_2d.squeeze(0).permute(1, 0, 2).unsqueeze(0)
                
                output, self.hidden_states[i] = model.forward_single_window(
                    window_2d, self.hidden_states[i]
                )
                all_predictions.append(output.numpy().flatten())
        
        if not data_point.need_prediction:
            return None
        
        # Average predictions from all models
        ensemble_prediction = np.mean(all_predictions, axis=0)
        return np.clip(ensemble_prediction, -6.0, 6.0)


if __name__ == "__main__":
    import sys
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
                        self.device = torch.device("cpu")
                        
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
        print("Testing exp_25b LOB-ViT Snapshot Ensemble...")
        model = PredictionModel()
        scorer = ScorerStepByStep("datasets/valid.parquet")
        results = scorer.score(model)
        
        print(f"\nResults:")
        print(results)
    print(results)