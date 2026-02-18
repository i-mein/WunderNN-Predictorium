"""
exp_24: LiT-inspired LOB Transformer - Submission Solution

Standalone inference solution for competition submission.
Uses LiTModelSingleStep for efficient step-by-step prediction.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import DataPoint from utils (provided by competition)
from utils import DataPoint


# ============================================
# Model Architecture (copied for standalone)
# ============================================
class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""
    
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        return x + self.pos_embedding(positions)


class CausalTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with causal attention mask."""
    
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
    """Stack of causal transformer encoder layers."""
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        seq_len = x.shape[1]
        attn_mask = self.generate_causal_mask(seq_len, x.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x


class LiTModelSingleStep(nn.Module):
    """LiT model optimized for single-step inference."""
    
    def __init__(
        self,
        input_dim=32,
        embed_dim=64,
        num_transformer_layers=2,
        num_heads=4,
        ff_dim=256,
        gru_hidden=128,
        gru_layers=2,
        output_dim=2,
        dropout=0.0,
        context_len=10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.context_len = context_len
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, context_len + 1)
        
        # Causal Transformer
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout
        )
        
        # GRU
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward(self, x, hidden=None, context_buffer=None):
        """Single-step forward pass."""
        batch = x.shape[0]
        
        # Project input
        x_embed = self.input_proj(x.unsqueeze(1))
        
        # Initialize context buffer if needed
        if context_buffer is None:
            context_buffer = torch.zeros(batch, self.context_len, self.embed_dim, device=x.device)
        
        # Build full context
        full_context = torch.cat([context_buffer, x_embed], dim=1)
        full_context = self.pos_encoding(full_context)
        
        # Transformer
        transformer_out = self.transformer(full_context)
        current_out = transformer_out[:, -1:, :]
        
        # GRU step
        gru_out, hidden = self.gru(current_out, hidden)
        
        # Predictions
        predictions = self.output_head(gru_out.squeeze(1))
        
        # Update context buffer (FIFO)
        new_context = torch.cat([context_buffer[:, 1:, :], x_embed], dim=1)
        
        return predictions, hidden, new_context


# ============================================
# Prediction Model (Competition Interface)
# ============================================
class PredictionModel:
    """
    LiT-inspired Transformer for LOB prediction.
    
    Implements the required interface:
    - __init__(): Load model weights
    - predict(data_point: DataPoint) -> np.ndarray | None
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.hidden = None
        self.context_buffer = None
        
        # Running statistics for normalization
        self.running_mean = None
        self.running_var = None
        self.step_count = 0
        
        # Determine model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "..", "..", "best_model.pth")
        
        self.device = torch.device("cpu")
        
        # Model configuration (must match training)
        self.model = LiTModelSingleStep(
            input_dim=32,
            embed_dim=64,
            num_transformer_layers=2,
            num_heads=4,
            ff_dim=256,
            gru_hidden=128,
            gru_layers=2,
            context_len=10,
            dropout=0.0  # No dropout at inference
        )
        
        try:
            # Load state dict from training model
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Map training model weights to single-step model
            # The architectures are compatible but named differently
            new_state_dict = self._convert_state_dict(state_dict)
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded LiT model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _convert_state_dict(self, state_dict):
        """Convert training model state dict to single-step model format."""
        new_dict = {}
        
        for key, value in state_dict.items():
            # Map patch_embed to input_proj
            if key.startswith('patch_embed.projection'):
                new_key = key.replace('patch_embed.projection', 'input_proj.0')
                new_dict[new_key] = value
            elif key.startswith('patch_embed.norm'):
                new_key = key.replace('patch_embed.norm', 'input_proj.1')
                new_dict[new_key] = value
            elif key == 'pos_encoding.pos_embedding.weight':
                # Truncate positional encoding to context_len + 1
                # Training model has (100, 64), we need (context_len+1, 64)
                new_dict[key] = value[:11, :]  # context_len=10, so 11 positions
            else:
                new_dict[key] = value
        
        return new_dict
    
    def _normalize_features(self, state):
        """Online normalization with running statistics."""
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.running_mean is None:
            self.running_mean = state.copy()
            self.running_var = np.ones_like(state)
            self.step_count = 1
            return np.zeros_like(state)
        
        # Update running statistics (exponential moving average)
        alpha = 0.01
        self.running_mean = (1 - alpha) * self.running_mean + alpha * state
        self.running_var = (1 - alpha) * self.running_var + alpha * (state - self.running_mean) ** 2
        self.step_count += 1
        
        # Normalize
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (state - self.running_mean) / std
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Generate predictions for the current timestep.
        
        Args:
            data_point: DataPoint with state (32 raw features)
        
        Returns:
            np.ndarray of shape (2,) with [t0, t1] predictions, or None
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None
            self.context_buffer = None
            self.running_mean = None
            self.running_var = None
            self.step_count = 0
        
        if self.model is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Normalize features
        normalized_state = self._normalize_features(data_point.state)
        
        # Prepare input tensor
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output, self.hidden, self.context_buffer = self.model(
                state_tensor, self.hidden, self.context_buffer
            )
        
        if not data_point.need_prediction:
            return None
        
        prediction = output.squeeze().numpy()
        return np.clip(prediction, -6.0, 6.0)


# ============================================
# Local Testing
# ============================================
if __name__ == "__main__":
    # Test with local evaluation
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    from src.utils import ScorerStepByStep
    
    print("Testing exp_24 LiT model...")
    
    model = PredictionModel()
    scorer = ScorerStepByStep("datasets/valid.parquet")
    results = scorer.score(model)
    
    print(f"\nResults:")
    print(f"  t0 correlation: {results['t0_corr']:.4f}")
    print(f"  t1 correlation: {results['t1_corr']:.4f}")
    print(f"  Weighted score: {results['weighted_score']:.4f}")
