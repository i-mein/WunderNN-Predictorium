"""
Dual Causal Transformer Models for exp_11b

Two separate transformers:
- CausalTransformerT0: Optimized for t0 (simpler)
- CausalTransformerT1: Optimized for t1 (deeper)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    Single-output Causal Transformer.
    Used for either t0 or t1 prediction.
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
        
        # Single output head
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            pred: (batch, seq_len, 1)
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
        
        # Prediction
        pred = self.head(x)
        
        return pred


class DualTransformerModel(nn.Module):
    """
    Wrapper that holds two separate transformer models for t0 and t1.
    """
    
    def __init__(self, model_t0_config, model_t1_config):
        super().__init__()
        
        self.model_t0 = CausalTransformerSingle(**model_t0_config)
        self.model_t1 = CausalTransformerSingle(**model_t1_config)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through both models.
        
        Args:
            x: (batch, seq_len, input_dim)
            hidden: Not used (API compatibility)
        
        Returns:
            pred_t0: (batch, seq_len, 1)
            pred_t1: (batch, seq_len, 1)
            hidden: None
        """
        pred_t0 = self.model_t0(x)
        pred_t1 = self.model_t1(x)
        
        return pred_t0, pred_t1, None


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import MODEL_T0_CONFIG, MODEL_T1_CONFIG
    
    # Test individual models
    model_t0 = CausalTransformerSingle(**MODEL_T0_CONFIG)
    model_t1 = CausalTransformerSingle(**MODEL_T1_CONFIG)
    
    print(f"Model T0 parameters: {count_parameters(model_t0):,}")
    print(f"Model T1 parameters: {count_parameters(model_t1):,}")
    
    # Test dual model
    dual_model = DualTransformerModel(MODEL_T0_CONFIG, MODEL_T1_CONFIG)
    print(f"Total parameters: {count_parameters(dual_model):,}")
    
    # Test forward pass
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, _ = dual_model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output t0 shape: {pred_t0.shape}")
    print(f"Output t1 shape: {pred_t1.shape}")
