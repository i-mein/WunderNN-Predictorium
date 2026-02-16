"""
exp_21: Multi-Stream GRU with Temporal Attention
Solution module for step-by-step inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path


class LeakageFreeFeatures(nn.Module):
    """
    Compute engineered features from CURRENT timestep only.
    """
    
    def forward(self, x):
        """
        x: (batch, seq, 32) raw features
        Returns: (batch, seq, 8) engineered features
        """
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
        
        mid_price = (p0 + p1) / 2
        spread = p1 - p0
        spread_pct = spread / (mid_price.abs() + eps)
        vol_imbalance = (v0 - v1) / (v0 + v1 + eps)
        best_vol_imbalance = v0 / (v0 + v1 + eps)
        price_momentum = dp0
        vol_momentum = dv0
        bid_vol = v0 + v2 + v4
        ask_vol = v1 + v3 + v5
        book_pressure = bid_vol / (ask_vol + eps)
        
        features = torch.stack([
            mid_price, spread, spread_pct, vol_imbalance,
            best_vol_imbalance, price_momentum, vol_momentum, book_pressure,
        ], dim=-1)
        
        return features


class TemporalAttention(nn.Module):
    """Attention over temporal dimension of GRU outputs."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, return_weights=False):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        if return_weights:
            return output, attn_weights
        return output


class StreamEncoderWithAttention(nn.Module):
    """Stream encoder with GRU followed by temporal attention."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.temporal_attn = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        gru_out, new_hidden = self.gru(x, hidden)
        attn_out = self.temporal_attn(gru_out)
        x = self.norm(gru_out + attn_out)
        output = self.output_proj(x)
        return output, new_hidden


class CrossStreamAttention(nn.Module):
    """Cross-attention between different feature streams. CAUSAL."""
    
    def __init__(self, dims, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.stream_names = list(dims.keys())
        total_dim = sum(dims.values())
        
        self.attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(total_dim)
        
    def forward(self, stream_outputs):
        concat = torch.cat([stream_outputs[name] for name in self.stream_names], dim=-1)
        # Create causal mask
        seq_len = concat.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=concat.device, dtype=torch.bool), 
            diagonal=1
        )
        attn_out, _ = self.attention(concat, concat, concat, attn_mask=causal_mask)
        output = self.norm(concat + attn_out)
        return output


class MultiStreamAttentionModel(nn.Module):
    """Multi-Stream GRU with Temporal Attention."""
    
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
        
        self.feature_eng = LeakageFreeFeatures()
        
        self.price_stream = StreamEncoderWithAttention(
            input_dim=12, hidden_dim=price_hidden, output_dim=price_output,
            num_layers=2, num_heads=num_attn_heads, dropout=dropout,
        )
        
        self.volume_stream = StreamEncoderWithAttention(
            input_dim=12, hidden_dim=volume_hidden, output_dim=volume_output,
            num_layers=2, num_heads=num_attn_heads, dropout=dropout,
        )
        
        self.dprice_stream = StreamEncoderWithAttention(
            input_dim=4, hidden_dim=delta_hidden, output_dim=delta_output,
            num_layers=1, num_heads=2, dropout=dropout,
        )
        
        self.dvolume_stream = StreamEncoderWithAttention(
            input_dim=4, hidden_dim=delta_hidden, output_dim=delta_output,
            num_layers=1, num_heads=2, dropout=dropout,
        )
        
        self.eng_stream = StreamEncoderWithAttention(
            input_dim=8, hidden_dim=eng_hidden, output_dim=eng_output,
            num_layers=1, num_heads=2, dropout=dropout,
        )
        
        stream_dims = {
            'price': price_output,
            'volume': volume_output,
            'dprice': delta_output,
            'dvolume': delta_output,
            'eng': eng_output,
        }
        
        self.cross_stream_attn = CrossStreamAttention(
            dims=stream_dims, num_heads=num_attn_heads, dropout=dropout,
        )
        
        combined_dim = sum(stream_dims.values())
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
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
        
        self.stream_dims = stream_dims
        
    def forward(self, x, hidden=None):
        prices = x[..., :12]
        volumes = x[..., 12:24]
        dprices = x[..., 24:28]
        dvolumes = x[..., 28:32]
        
        eng_features = self.feature_eng(x)
        
        if hidden is None:
            hidden = {
                'price': None, 'volume': None, 'dprice': None,
                'dvolume': None, 'eng': None,
            }
        
        price_out, hidden['price'] = self.price_stream(prices, hidden['price'])
        volume_out, hidden['volume'] = self.volume_stream(volumes, hidden['volume'])
        dprice_out, hidden['dprice'] = self.dprice_stream(dprices, hidden['dprice'])
        dvolume_out, hidden['dvolume'] = self.dvolume_stream(dvolumes, hidden['dvolume'])
        eng_out, hidden['eng'] = self.eng_stream(eng_features, hidden['eng'])
        
        stream_outputs = {
            'price': price_out, 'volume': volume_out,
            'dprice': dprice_out, 'dvolume': dvolume_out, 'eng': eng_out,
        }
        combined = self.cross_stream_attn(stream_outputs)
        
        fused = self.fusion(combined)
        
        pred_t0 = self.t0_head(fused).squeeze(-1)
        pred_t1 = self.t1_head(fused).squeeze(-1)
        
        return pred_t0, pred_t1, hidden


class PredictionModel:
    """
    Step-by-step prediction model for evaluation.
    """
    
    def __init__(self):
        # Find model file
        solution_dir = Path(__file__).parent
        model_path = solution_dir / 'best_model.pth'
        
        # Set device
        self.device = torch.device('cpu')  # Use CPU for evaluation
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = MultiStreamAttentionModel(
            price_hidden=config.get('price_hidden', 48),
            price_output=config.get('price_output', 64),
            volume_hidden=config.get('volume_hidden', 48),
            volume_output=config.get('volume_output', 64),
            delta_hidden=config.get('delta_hidden', 32),
            delta_output=config.get('delta_output', 32),
            eng_hidden=config.get('eng_hidden', 32),
            eng_output=config.get('eng_output', 48),
            head_hidden=config.get('head_hidden', 128),
            dropout=0.0,  # No dropout at inference
            num_attn_heads=config.get('num_attn_heads', 4),
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # State
        self.hidden = None
        self.current_stock_id = None
        self.current_day = None
        
        print(f"Loaded exp_21 model from {model_path}")
        print(f"  Valid corr avg: {checkpoint.get('valid_corr_avg', 'N/A'):.4f}")
    
    def reset_hidden(self):
        """Reset hidden state for new sequence."""
        self.hidden = None
    
    @torch.no_grad()
    def predict(self, features: np.ndarray, stock_id: int, day: int) -> tuple:
        """
        Predict for a single timestep.
        
        Args:
            features: (32,) array of features
            stock_id: stock identifier
            day: day identifier
            
        Returns:
            (pred_t0, pred_t1) tuple of predictions
        """
        # Reset hidden state if stock/day changed
        if stock_id != self.current_stock_id or day != self.current_day:
            self.reset_hidden()
            self.current_stock_id = stock_id
            self.current_day = day
        
        # Prepare input: (1, 1, 32)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        pred_t0, pred_t1, self.hidden = self.model(x, self.hidden)
        
        return pred_t0.item(), pred_t1.item()
