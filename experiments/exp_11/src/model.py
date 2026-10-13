"""
Causal Transformer Model for exp_11

Based on "Transformers for Limit Order Books" (arXiv:2003.00130)
Key: Causal masking in self-attention prevents future leakage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position awareness.
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    
    Causal mask ensures position i can only attend to positions <= i.
    This prevents information leakage from future timesteps.
    """
    
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
        
        # Causal mask will be created dynamically based on sequence length
        self.register_buffer('mask', None)
    
    def _get_causal_mask(self, seq_len, device):
        """Create causal mask: upper triangular = -inf"""
        # mask[i,j] = -inf if j > i (can't attend to future)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, n_heads, seq, seq)
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq, d_k)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(context)
        
        return output


class TransformerBlock(nn.Module):
    """
    Single Transformer block with causal self-attention.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Causal self-attention
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
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
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CausalTransformer(nn.Module):
    """
    Causal Transformer for LOB prediction.
    
    Architecture:
    - Input projection: 32 -> d_model
    - Positional encoding
    - N Transformer blocks with causal attention
    - Dual heads for t0/t1
    """
    
    def __init__(
        self,
        input_dim=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        head_hidden=64,
        dropout=0.3,
        dropout_t0=0.2,
        dropout_t1=0.4,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dual heads
        self.t0_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t0),
            nn.Linear(head_hidden, 1)
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_t1),
            nn.Linear(head_hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            hidden: Not used (for API compatibility with GRU models)
        
        Returns:
            pred_t0: (batch, seq_len, 1)
            pred_t1: (batch, seq_len, 1)
            hidden: None (no hidden state in Transformer)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Output norm
        x = self.output_norm(x)
        
        # Dual heads
        pred_t0 = self.t0_head(x)
        pred_t1 = self.t1_head(x)
        
        return pred_t0, pred_t1, None


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CausalTransformer()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 100, 32)  # (batch=2, seq=100, features=32)
    pred_t0, pred_t1, _ = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output t0 shape: {pred_t0.shape}")
    print(f"Output t1 shape: {pred_t1.shape}")
