"""
exp_21: Multi-Stream GRU with Temporal Attention
=================================================

Building on exp_18's multi-stream architecture with:
1. Temporal attention over GRU hidden states (not just final output)
2. More careful leakage-free feature engineering with explicit indexing
3. Cross-stream attention for stream interaction

Architecture:
- 5 streams: prices, volumes, delta_prices, delta_volumes, engineered
- Each stream has GRU + temporal attention over sequence
- Cross-stream attention for interactions
- Separate heads for t0 and t1

Key differences from exp_18:
- Temporal attention captures important timesteps per stream
- Cross-stream attention lets streams attend to each other
- More robust leakage-free features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeakageFreeFeatures(nn.Module):
    """
    Compute engineered features from CURRENT timestep only.
    All operations use explicit slicing to avoid look-ahead.
    """
    
    def forward(self, x):
        """
        x: (batch, seq, 32) raw features
        Returns: (batch, seq, 8) engineered features
        
        Features (all computed from current timestep):
        - mid_price: (p0 + p1) / 2
        - spread: p1 - p0 (ask - bid)
        - spread_pct: spread / mid_price
        - vol_imbalance: (v0 - v1) / (v0 + v1 + eps)
        - best_vol_imbalance: v0 / (v0 + v1 + eps)
        - price_momentum: dp0 (already provided)
        - vol_momentum: dv0 (already provided)
        - book_pressure: (v0 + v2 + v4) / (v1 + v3 + v5 + eps)
        """
        # Extract features using explicit indices
        p0 = x[..., 0]   # best bid
        p1 = x[..., 1]   # best ask
        v0 = x[..., 12]  # bid volume level 0
        v1 = x[..., 13]  # ask volume level 0
        v2 = x[..., 14]  # bid volume level 1
        v3 = x[..., 15]  # ask volume level 1
        v4 = x[..., 16]  # bid volume level 2
        v5 = x[..., 17]  # ask volume level 2
        dp0 = x[..., 24] # delta price 0
        dv0 = x[..., 28] # delta volume 0
        
        eps = 1e-8
        
        # Mid price
        mid_price = (p0 + p1) / 2
        
        # Spread (ask - bid)
        spread = p1 - p0
        
        # Spread percentage
        spread_pct = spread / (mid_price.abs() + eps)
        
        # Volume imbalance (bid - ask normalized)
        vol_imbalance = (v0 - v1) / (v0 + v1 + eps)
        
        # Best bid volume ratio
        best_vol_imbalance = v0 / (v0 + v1 + eps)
        
        # Price momentum (from deltas)
        price_momentum = dp0
        
        # Volume momentum (from deltas)
        vol_momentum = dv0
        
        # Book pressure (bid side / ask side for top 3 levels)
        bid_vol = v0 + v2 + v4
        ask_vol = v1 + v3 + v5
        book_pressure = bid_vol / (ask_vol + eps)
        
        # Stack features
        features = torch.stack([
            mid_price,
            spread,
            spread_pct,
            vol_imbalance,
            best_vol_imbalance,
            price_momentum,
            vol_momentum,
            book_pressure,
        ], dim=-1)
        
        return features


class TemporalAttention(nn.Module):
    """
    Attention over temporal dimension of GRU outputs.
    Learns to weight different timesteps based on importance.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, return_weights=False):
        """
        x: (batch, seq, hidden_dim)
        Returns: (batch, seq, hidden_dim) attended outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask (only attend to current and past)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if return_weights:
            return output, attn_weights
        return output


class StreamEncoderWithAttention(nn.Module):
    """
    Stream encoder with GRU followed by temporal attention.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer norm after attention
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq, input_dim)
        hidden: GRU hidden state
        
        Returns: (batch, seq, output_dim), new_hidden
        """
        # Project input
        x = self.input_proj(x)
        
        # GRU
        gru_out, new_hidden = self.gru(x, hidden)
        
        # Temporal attention (residual connection)
        attn_out = self.temporal_attn(gru_out)
        x = self.norm(gru_out + attn_out)
        
        # Output projection
        output = self.output_proj(x)
        
        return output, new_hidden


class CrossStreamAttention(nn.Module):
    """
    Cross-attention between different feature streams.
    Allows streams to exchange information.
    CAUSAL: Each timestep can only attend to current and past timesteps.
    """
    
    def __init__(self, dims, num_heads=4, dropout=0.1):
        """
        dims: dict mapping stream name to dimension
        """
        super().__init__()
        
        self.stream_names = list(dims.keys())
        total_dim = sum(dims.values())
        
        # Attention over concatenated streams
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(total_dim)
        
    def forward(self, stream_outputs):
        """
        stream_outputs: dict mapping stream name to (batch, seq, dim) tensor
        Returns: fused output (batch, seq, total_dim)
        """
        # Concatenate all streams
        concat = torch.cat([stream_outputs[name] for name in self.stream_names], dim=-1)
        
        # Create causal mask (upper triangular = True means masked/ignored)
        seq_len = concat.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=concat.device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Self-attention over concatenated streams WITH CAUSAL MASK
        attn_out, _ = self.attention(concat, concat, concat, attn_mask=causal_mask)
        
        # Residual + norm
        output = self.norm(concat + attn_out)
        
        return output


class MultiStreamAttentionModel(nn.Module):
    """
    Multi-Stream GRU with Temporal Attention.
    
    Streams:
    - prices: p0-p11 (12 features) -> 64 dim
    - volumes: v0-v11 (12 features) -> 64 dim  
    - delta_prices: dp0-dp3 (4 features) -> 32 dim
    - delta_volumes: dv0-dv3 (4 features) -> 32 dim
    - engineered: 8 leakage-free features -> 48 dim
    
    Total: 240 dim
    """
    
    def __init__(
        self,
        price_hidden=48, price_output=64,
        volume_hidden=48, volume_output=64,
        delta_hidden=32, delta_output=32,
        eng_hidden=32, eng_output=48,
        head_hidden=128,
        dropout=0.2,
        num_attn_heads=4,
    ):
        super().__init__()
        
        # Leakage-free feature engineering
        self.feature_eng = LeakageFreeFeatures()
        
        # Stream encoders with temporal attention
        self.price_stream = StreamEncoderWithAttention(
            input_dim=12,
            hidden_dim=price_hidden,
            output_dim=price_output,
            num_layers=2,
            num_heads=num_attn_heads,
            dropout=dropout,
        )
        
        self.volume_stream = StreamEncoderWithAttention(
            input_dim=12,
            hidden_dim=volume_hidden,
            output_dim=volume_output,
            num_layers=2,
            num_heads=num_attn_heads,
            dropout=dropout,
        )
        
        self.dprice_stream = StreamEncoderWithAttention(
            input_dim=4,
            hidden_dim=delta_hidden,
            output_dim=delta_output,
            num_layers=1,
            num_heads=2,  # Fewer heads for smaller dim
            dropout=dropout,
        )
        
        self.dvolume_stream = StreamEncoderWithAttention(
            input_dim=4,
            hidden_dim=delta_hidden,
            output_dim=delta_output,
            num_layers=1,
            num_heads=2,
            dropout=dropout,
        )
        
        # Engineered features - lighter processing with attention
        self.eng_stream = StreamEncoderWithAttention(
            input_dim=8,  # 8 engineered features
            hidden_dim=eng_hidden,
            output_dim=eng_output,
            num_layers=1,
            num_heads=2,
            dropout=dropout,
        )
        
        # Cross-stream attention
        stream_dims = {
            'price': price_output,
            'volume': volume_output,
            'dprice': delta_output,
            'dvolume': delta_output,
            'eng': eng_output,
        }
        combined_dim = sum(stream_dims.values())  # 240
        
        self.cross_stream_attn = CrossStreamAttention(
            dims=stream_dims,
            num_heads=num_attn_heads,
            dropout=dropout,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Prediction heads
        self.t0_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        # Store for hidden state management
        self.stream_dims = stream_dims
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq, 32) raw features
        hidden: dict of hidden states per stream
        
        Returns: pred_t0, pred_t1, hidden
        """
        # Split raw features into streams
        prices = x[..., :12]       # p0-p11
        volumes = x[..., 12:24]    # v0-v11
        dprices = x[..., 24:28]    # dp0-dp3
        dvolumes = x[..., 28:32]   # dv0-dv3
        
        # Compute engineered features
        eng_features = self.feature_eng(x)  # (batch, seq, 8)
        
        # Initialize hidden states if needed
        if hidden is None:
            hidden = {
                'price': None,
                'volume': None,
                'dprice': None,
                'dvolume': None,
                'eng': None,
            }
        
        # Process each stream
        price_out, hidden['price'] = self.price_stream(prices, hidden['price'])
        volume_out, hidden['volume'] = self.volume_stream(volumes, hidden['volume'])
        dprice_out, hidden['dprice'] = self.dprice_stream(dprices, hidden['dprice'])
        dvolume_out, hidden['dvolume'] = self.dvolume_stream(dvolumes, hidden['dvolume'])
        eng_out, hidden['eng'] = self.eng_stream(eng_features, hidden['eng'])
        
        # Cross-stream attention
        stream_outputs = {
            'price': price_out,
            'volume': volume_out,
            'dprice': dprice_out,
            'dvolume': dvolume_out,
            'eng': eng_out,
        }
        combined = self.cross_stream_attn(stream_outputs)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Predictions
        pred_t0 = self.t0_head(fused).squeeze(-1)
        pred_t1 = self.t1_head(fused).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


if __name__ == "__main__":
    # Test model
    model = MultiStreamAttentionModel()
    
    print("=" * 60)
    print("Multi-Stream Attention Model (exp_21)")
    print("=" * 60)
    
    print("\nTraining mode (full sequence):")
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  pred_t0 shape: {pred_t0.shape}")
    print(f"  pred_t1 shape: {pred_t1.shape}")
    
    print("\nInference mode (single step):")
    x_step = torch.randn(1, 1, 32)
    pred_t0_step, pred_t1_step, hidden_step = model(x_step, hidden=None)
    print(f"  Input shape: {x_step.shape}")
    print(f"  pred_t0 shape: {pred_t0_step.shape}")
    print(f"  pred_t1 shape: {pred_t1_step.shape}")
    
    print("\nInference mode (with carried hidden state):")
    pred_t0_step2, pred_t1_step2, hidden_step2 = model(x_step, hidden=hidden_step)
    print(f"  Hidden state carried successfully")
    
    print("\n" + "=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
