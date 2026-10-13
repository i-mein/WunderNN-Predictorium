"""
Exp 13: Hybrid Transformer-GRU Model

Key insights from data analysis:
1. dp1 is the most important feature for t1 (0.149 importance)
2. Delta features (dp0-dp3, dv0-dv3) are crucial for t1
3. t1 has weaker correlations with features but higher autocorrelation
4. Price features (p0, p8, v8) are key for t0

Architecture:
- Feature engineering layer (mid-price, spread, volume imbalance)
- Transformer blocks for local attention patterns
- GRU for temporal memory (especially using first 100 steps)
- Separate prediction heads with different complexity for t0 vs t1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEngineer(nn.Module):
    """
    Computes engineered features based on analysis findings:
    - Mid-price: (p0 + p6) / 2
    - Spread: p6 - p0
    - Volume imbalance at each level
    - Price momentum (dp0 + dp1 + dp2 + dp3)
    - Volume momentum (dv0 + dv1 + dv2 + dv3)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: (batch, seq, 32) - original features
        Returns: (batch, seq, 32 + n_engineered)
        """
        # Original features layout:
        # p0-p11 (indices 0-11): prices
        # v0-v11 (indices 12-23): volumes
        # dp0-dp3 (indices 24-27): delta prices
        # dv0-dv3 (indices 28-31): delta volumes
        
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        # Best bid/ask
        p0, p6 = prices[..., 0:1], prices[..., 6:7]
        v0, v6 = volumes[..., 0:1], volumes[..., 6:7]
        
        # Mid-price
        mid_price = (p0 + p6) / 2
        
        # Spread
        spread = p6 - p0
        
        # Volume imbalance (bid vs ask at each level)
        bid_vol = volumes[..., :6]  # v0-v5
        ask_vol = volumes[..., 6:]  # v6-v11
        vol_imbalance = (bid_vol.sum(dim=-1, keepdim=True) - ask_vol.sum(dim=-1, keepdim=True)) / \
                        (bid_vol.sum(dim=-1, keepdim=True) + ask_vol.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Best level volume imbalance
        best_vol_imbalance = (v0 - v6) / (v0 + v6 + 1e-8)
        
        # Price momentum (sum of deltas)
        price_momentum = dprices.sum(dim=-1, keepdim=True)
        
        # Volume momentum
        vol_momentum = dvolumes.sum(dim=-1, keepdim=True)
        
        # Weighted price (by volume at each level)
        weighted_bid = (prices[..., :6] * volumes[..., :6]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., :6].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_ask = (prices[..., 6:] * volumes[..., 6:]).sum(dim=-1, keepdim=True) / \
                       (volumes[..., 6:].sum(dim=-1, keepdim=True) + 1e-8)
        weighted_mid = (weighted_bid + weighted_ask) / 2
        
        # Stack engineered features
        engineered = torch.cat([
            mid_price,           # 1
            spread,              # 1
            vol_imbalance,       # 1
            best_vol_imbalance,  # 1
            price_momentum,      # 1
            vol_momentum,        # 1
            weighted_mid,        # 1
        ], dim=-1)  # Total: 7 new features
        
        # Concatenate with original
        return torch.cat([x, engineered], dim=-1)  # 32 + 7 = 39


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=1100, dropout=0.1):
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
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class HybridTransformerGRU(nn.Module):
    """
    Hybrid model combining:
    1. Feature engineering layer
    2. Transformer blocks for local attention
    3. Bidirectional GRU for temporal patterns
    4. Separate heads for t0 and t1
    
    Key design choices:
    - Use warmup steps (0-99) to build GRU hidden state
    - Transformer captures local attention patterns
    - GRU captures long-term temporal dependencies
    - t1 head is deeper (uses more delta features emphasized)
    """
    
    def __init__(
        self,
        input_dim=32,
        d_model=128,
        n_heads=4,
        n_transformer_layers=2,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        engineered_dim = input_dim + 7  # 39 total
        
        # Input projection
        self.input_proj = nn.Linear(engineered_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
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
            bidirectional=False,  # Causal - can only look at past
        )
        
        # Combine transformer + GRU outputs
        combined_dim = d_model + gru_hidden
        
        # Separate prediction heads
        # t0 head - simpler (prices are more important)
        self.t0_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # t1 head - deeper (delta features are key, needs more capacity)
        self.t1_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
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
        
        # Delta features attention for t1 (emphasize dp1, dv features)
        self.delta_attn = nn.Sequential(
            nn.Linear(8, 32),  # 4 dp + 4 dv
            nn.GELU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, 32) original features
        hidden: GRU hidden state
        
        Returns: pred_t0, pred_t1, hidden
        """
        batch_size, seq_len, _ = x.shape
        
        # Engineer features
        x_eng = self.feature_engineer(x)  # (B, T, 39)
        
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
        
        # Predictions
        pred_t0 = self.t0_head(combined).squeeze(-1)  # (B, T)
        pred_t1 = self.t1_head(combined).squeeze(-1)  # (B, T)
        
        return pred_t0, pred_t1, hidden
    
    def forward_step(self, x, hidden=None):
        """
        Single step forward for inference.
        x: (batch, 1, 32) single timestep
        """
        return self.forward(x, hidden)


class HybridModel(nn.Module):
    """
    Wrapper for training/inference compatibility.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = HybridTransformerGRU(**kwargs)
    
    def forward(self, x, hidden=None):
        pred_t0, pred_t1, hidden = self.model(x, hidden)
        # Stack predictions for compatibility
        predictions = torch.stack([pred_t0, pred_t1], dim=-1)  # (B, T, 2)
        return predictions, hidden


if __name__ == "__main__":
    # Test model
    model = HybridTransformerGRU()
    
    # Test 1: Full sequence (training mode)
    print("=" * 50)
    print("Training mode (full sequence):")
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  pred_t0 shape: {pred_t0.shape} (B, T)")
    print(f"  pred_t1 shape: {pred_t1.shape} (B, T)")
    print(f"  hidden shape: {hidden.shape}")
    
    # Test 2: Single step (inference mode)
    print("\n" + "=" * 50)
    print("Inference mode (single step):")
    x_step = torch.randn(1, 1, 32)
    pred_t0_step, pred_t1_step, hidden_step = model(x_step, hidden=None)
    print(f"  Input shape: {x_step.shape}")
    print(f"  pred_t0 shape: {pred_t0_step.shape} -> last step: {pred_t0_step[:, -1:].shape} (B, 1)")
    print(f"  pred_t1 shape: {pred_t1_step.shape} -> last step: {pred_t1_step[:, -1:].shape} (B, 1)")
    
    # Count parameters
    print("\n" + "=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
