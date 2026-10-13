"""
Exp 14: Enhanced Hybrid Transformer-GRU Model

Improvements over exp_13:
1. Deeper transformer (3 layers instead of 2)
2. Larger GRU hidden size (192 vs 128)
3. Dedicated delta feature path for t1
4. Layer-wise learning rate decay
5. Pre-normalization in transformer blocks

Key insight from data analysis:
- dp1 is the most important feature for t1 (0.149 importance)
- Delta features (dp0-dp3, dv0-dv3) are crucial for t1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEngineer(nn.Module):
    """
    Enhanced feature engineering with delta feature emphasis.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: (batch, seq, 32) - original features
        Returns: (batch, seq, 32 + n_engineered)
        """
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        p0, p6 = prices[..., 0:1], prices[..., 6:7]
        v0, v6 = volumes[..., 0:1], volumes[..., 6:7]
        
        # Mid-price
        mid_price = (p0 + p6) / 2
        
        # Spread
        spread = p6 - p0
        
        # Volume imbalance
        bid_vol = volumes[..., :6]
        ask_vol = volumes[..., 6:]
        vol_imbalance = (bid_vol.sum(dim=-1, keepdim=True) - ask_vol.sum(dim=-1, keepdim=True)) / \
                        (bid_vol.sum(dim=-1, keepdim=True) + ask_vol.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Best level volume imbalance
        best_vol_imbalance = (v0 - v6) / (v0 + v6 + 1e-8)
        
        # Price momentum (sum of deltas)
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        
        # Volume momentum
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        # Weighted mid-price
        weighted_bid = (prices[..., :6] * volumes[..., :6]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., :6].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_ask = (prices[..., 6:] * volumes[..., 6:]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., 6:].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_mid = (weighted_bid + weighted_ask) / 2
        
        # NEW: Delta feature interactions (key for t1)
        dp1_squared = dprices[..., 1:2] ** 2  # dp1 is most important for t1
        dp_dv_interaction = dprices[..., 1:2] * dvolumes[..., 1:2]  # dp1 * dv1
        delta_ratio = dprices.sum(dim=-1, keepdim=True) / (dvolumes.sum(dim=-1, keepdim=True).abs() + 1e-8)
        
        engineered = torch.cat([
            mid_price,           # 1
            spread,              # 1
            vol_imbalance,       # 1
            best_vol_imbalance,  # 1
            price_momentum,      # 1
            vol_momentum,        # 1
            weighted_mid,        # 1
            dp1_squared,         # 1 - NEW
            dp_dv_interaction,   # 1 - NEW
            delta_ratio,         # 1 - NEW
        ], dim=-1)  # Total: 10 new features
        
        return torch.cat([x, engineered], dim=-1)  # 32 + 10 = 42


class PositionalEncoding(nn.Module):
    """Learnable positional encoding with sinusoidal initialization."""
    
    def __init__(self, d_model, max_len=1100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize with sinusoidal, but make learnable
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=True)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and rotary embeddings."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with gated FFN."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        # Gated FFN (SwiGLU-style)
        self.ffn_gate = nn.Linear(d_model, d_ff)
        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_down = nn.Linear(d_ff, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm attention
        x = x + self.attn(self.norm1(x), mask)
        
        # Pre-norm gated FFN
        h = self.norm2(x)
        gate = F.silu(self.ffn_gate(h))
        up = self.ffn_up(h)
        x = x + self.ffn_dropout(self.ffn_down(gate * up))
        
        return x


class DeltaFeatureProcessor(nn.Module):
    """
    Dedicated processor for delta features (key for t1 prediction).
    Uses 1D convolutions to capture local patterns in deltas.
    """
    
    def __init__(self, delta_dim=8, hidden_dim=64, output_dim=32):
        super().__init__()
        
        self.conv1 = nn.Conv1d(delta_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        self.activation = nn.GELU()
    
    def forward(self, delta_features):
        """
        delta_features: (B, T, 8) - dp0-dp3, dv0-dv3
        Returns: (B, T, output_dim)
        """
        # (B, T, C) -> (B, C, T) for conv1d
        x = delta_features.transpose(1, 2)
        
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        
        x = self.conv3(x)
        x = x.transpose(1, 2)
        x = self.norm3(x)
        
        return x  # (B, T, output_dim)


class EnhancedHybridModel(nn.Module):
    """
    Enhanced Hybrid Transformer-GRU with:
    1. Deeper transformer (3 layers)
    2. Larger GRU (192 hidden)
    3. Dedicated delta feature processor for t1
    4. Gated FFN in transformer
    5. Learnable positional encoding
    """
    
    def __init__(
        self,
        input_dim=32,
        d_model=160,
        n_heads=8,
        n_transformer_layers=3,
        gru_hidden=192,
        gru_layers=2,
        dropout=0.15,
    ):
        super().__init__()
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        engineered_dim = input_dim + 10  # 42 total
        
        # Input projection
        self.input_proj = nn.Linear(engineered_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_transformer_layers)
        ])
        
        # GRU for temporal patterns
        self.gru = nn.GRU(
            d_model,
            gru_hidden,
            gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Delta feature processor for t1
        self.delta_processor = DeltaFeatureProcessor(
            delta_dim=8,
            hidden_dim=64,
            output_dim=48
        )
        
        # Combine transformer + GRU outputs
        combined_dim = d_model + gru_hidden
        
        # t0 head - uses main features (prices are most important)
        self.t0_head = nn.Sequential(
            nn.Linear(combined_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        
        # t1 head - uses main features + delta processor output
        t1_input_dim = combined_dim + 48  # 48 from delta processor
        self.t1_head = nn.Sequential(
            nn.Linear(t1_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.d_model = d_model
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, 32) original features
        hidden: GRU hidden state
        
        Returns: pred_t0, pred_t1, hidden
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract delta features before engineering
        delta_features = x[..., 24:32]  # dp0-dp3, dv0-dv3
        
        # Engineer features
        x_eng = self.feature_engineer(x)  # (B, T, 42)
        
        # Project to d_model
        h = self.input_proj(x_eng)
        h = self.input_norm(h)
        h = self.pos_encoder(h)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            h = block(h)
        
        transformer_out = h  # (B, T, d_model)
        
        # GRU
        if hidden is None:
            hidden = torch.zeros(
                self.gru_layers, batch_size, self.gru_hidden,
                device=x.device
            )
        
        gru_out, hidden = self.gru(transformer_out, hidden)  # (B, T, gru_hidden)
        
        # Combine transformer and GRU outputs
        combined = torch.cat([transformer_out, gru_out], dim=-1)  # (B, T, d_model + gru_hidden)
        
        # Process delta features for t1
        delta_processed = self.delta_processor(delta_features)  # (B, T, 48)
        
        # t0 prediction (main features)
        pred_t0 = self.t0_head(combined).squeeze(-1)  # (B, T)
        
        # t1 prediction (main features + delta processed)
        t1_input = torch.cat([combined, delta_processed], dim=-1)
        pred_t1 = self.t1_head(t1_input).squeeze(-1)  # (B, T)
        
        return pred_t0, pred_t1, hidden


if __name__ == "__main__":
    # Test model
    model = EnhancedHybridModel()
    
    print("=" * 50)
    print("Training mode (full sequence):")
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  pred_t0 shape: {pred_t0.shape} (B, T)")
    print(f"  pred_t1 shape: {pred_t1.shape} (B, T)")
    print(f"  hidden shape: {hidden.shape}")
    
    print("\n" + "=" * 50)
    print("Inference mode (single step):")
    x_step = torch.randn(1, 1, 32)
    pred_t0_step, pred_t1_step, hidden_step = model(x_step, hidden=None)
    print(f"  Input shape: {x_step.shape}")
    print(f"  pred_t0 shape: {pred_t0_step.shape} -> (B, 1)")
    print(f"  pred_t1 shape: {pred_t1_step.shape} -> (B, 1)")
    
    print("\n" + "=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
