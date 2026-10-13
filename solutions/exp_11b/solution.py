"""
Solution for exp_11b: Dual Causal Transformers

Two separate transformer models:
- Model T0: Optimized for t0 (simpler architecture)
- Model T1: Optimized for t1 (deeper architecture)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "src"))

try:
    from src.utils import DataPoint, ScorerStepByStep
except ImportError:
    from utils import DataPoint, ScorerStepByStep


# --- MODEL DEFINITIONS (Copied for standalone submission) ---

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        causal_mask = self._get_causal_mask(seq_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(context)
        return output


class TransformerBlock(nn.Module):
    """Single Transformer block with causal self-attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CausalTransformerSingle(nn.Module):
    """Single-output Causal Transformer."""
    
    def __init__(
        self,
        input_dim=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        head_hidden=64,
        dropout=0.3,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_norm(x)
        pred = self.head(x)
        
        return pred


# Model configs (must match training)
MODEL_T0_CONFIG = {
    'input_dim': 32,
    'd_model': 48,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 96,
    'head_hidden': 32,
    'dropout': 0.2,
}

MODEL_T1_CONFIG = {
    'input_dim': 32,
    'd_model': 96,
    'n_heads': 6,
    'n_layers': 4,
    'd_ff': 256,
    'head_hidden': 64,
    'dropout': 0.4,
}


class PredictionModel:
    """
    Exp 11b: Dual Causal Transformers for LOB prediction.
    
    Two separate models with different hyperparameters:
    - Model T0: Simpler (for easier t0 prediction)
    - Model T1: Deeper (for harder t1 prediction)
    """
    
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.history = []
        
        # Determine model paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_t0_path = os.path.join(base_dir, "best_model_t0.pth")
        model_t1_path = os.path.join(base_dir, "best_model_t1.pth")
        
        self.device = torch.device("cpu")
        
        # Initialize models
        self.model_t0 = CausalTransformerSingle(**MODEL_T0_CONFIG)
        self.model_t1 = CausalTransformerSingle(**MODEL_T1_CONFIG)
        
        # Load weights for T0
        try:
            checkpoint_t0 = torch.load(model_t0_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint_t0, dict) and 'model_state_dict' in checkpoint_t0:
                self.model_t0.load_state_dict(checkpoint_t0['model_state_dict'])
            else:
                self.model_t0.load_state_dict(checkpoint_t0)
            print(f"Loaded T0 model from: {model_t0_path}")
        except Exception as e:
            print(f"Error loading T0 model: {e}")
            self.model_t0 = None
        
        # Load weights for T1
        try:
            checkpoint_t1 = torch.load(model_t1_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint_t1, dict) and 'model_state_dict' in checkpoint_t1:
                self.model_t1.load_state_dict(checkpoint_t1['model_state_dict'])
            else:
                self.model_t1.load_state_dict(checkpoint_t1)
            print(f"Loaded T1 model from: {model_t1_path}")
        except Exception as e:
            print(f"Error loading T1 model: {e}")
            self.model_t1 = None
        
        # Set eval mode
        if self.model_t0:
            self.model_t0.to(self.device)
            self.model_t0.eval()
        if self.model_t1:
            self.model_t1.to(self.device)
            self.model_t1.eval()
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Predict for a single data point.
        
        Accumulates sequence history and processes full sequence
        with causal masking (each model sees only past data).
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []
        
        # Add current state to history
        self.history.append(data_point.state.astype(np.float32))
        
        if self.model_t0 is None or self.model_t1 is None:
            return np.zeros(2) if data_point.need_prediction else None
        
        # Prepare input: full sequence
        x = np.stack(self.history, axis=0)  # (seq_len, 32)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, seq_len, 32)
        
        # Inference
        with torch.no_grad():
            pred_t0 = self.model_t0(x)
            pred_t1 = self.model_t1(x)
        
        if not data_point.need_prediction:
            return None
        
        # Get prediction for last timestep
        t0 = pred_t0[0, -1, 0].cpu().item()
        t1 = pred_t1[0, -1, 0].cpu().item()
        
        return np.array([t0, t1], dtype=np.float32)


if __name__ == "__main__":
    # Local testing
    test_file = os.path.join(CURRENT_DIR, "..", "..", "datasets", "valid.parquet")
    
    if os.path.exists(test_file):
        print("Testing exp_11b solution...")
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
        print("Please run from solutions/exp_11b/ directory or ensure datasets are available")
