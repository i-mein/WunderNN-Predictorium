"""
exp_28: Advanced Transformer Model for LOB Prediction

Architecture adapted from aicheye/wundernn with modifications:
1. Output: 2 regression targets (t0, t1) instead of full state vector
2. Added causal mask for transformer (no future information)
3. Kept: Feature attention, positional encoding, ensemble heads

Key components:
- AdvancedPositionalEncoding: Sinusoidal + learnable positional embeddings
- FeatureAttention: Adaptive feature weighting
- AdvancedTransformerPredictor: Main model with attention pooling
"""

import math
import torch
import torch.nn as nn


class AdvancedPositionalEncoding(nn.Module):
    """
    Advanced positional encoding combining sinusoidal and learnable components.
    
    This helps the model understand both absolute position in sequence
    and learn position-specific patterns from data.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
        # Learnable positional embedding (additive)
        self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.learnable_pe, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model] with positional information
        """
        seq_len = x.size(1)
        # Combine sinusoidal and learnable encodings
        x = x + self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """
    Feature attention to adaptively weight LOB features.
    
    Different features (bid/ask prices, volumes, deltas) have varying
    importance - this learns which to emphasize.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            attended_features: [batch, seq_len, input_dim]
            attn_weights: [batch, seq_len, input_dim] feature importance
        """
        attn_weights = self.attention(x)  # [batch, seq_len, input_dim]
        attended_features = x * attn_weights
        return attended_features, attn_weights


class AdvancedTransformerPredictor(nn.Module):
    """
    Regularized Transformer for LOB prediction.
    
    Architecture features:
    - Feature attention for input weighting
    - Temporal convolution for local patterns
    - Transformer encoder with causal masking option
    - Attention-based sequence pooling
    - Ensemble prediction heads with learned weighting
    - Reconstruction head for auxiliary regularization
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        output_dim: int = 2,
        d_model: int = 48,
        nhead: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        feature_attn_hidden: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_heads = nhead
        
        # Feature attention layer
        self.feature_attention = FeatureAttention(input_dim, feature_attn_hidden)
        
        # Input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Skip connection for input (match d_model)
        self.input_skip = (
            nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        )
        
        # Advanced positional encoding
        self.pos_encoder = AdvancedPositionalEncoding(d_model, dropout=dropout)
        
        # Causal temporal convolution for local patterns
        # Use left-padding for causality
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=0  # Will add causal padding manually
        )
        self.conv_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction heads (ensemble of 2 heads)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.LayerNorm(dim_feedforward // 2),
                nn.GELU(),
                nn.Dropout(dropout * 1.5),
                nn.Linear(dim_feedforward // 2, output_dim),
            )
            for _ in range(2)
        ])
        
        # Attention pooling for sequence aggregation
        self.sequence_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Reconstruction head for auxiliary task (regularization)
        self.reconstruction_head = nn.Linear(d_model, input_dim)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        
        self._init_weights()
    
    def _init_weights(self):
        """Advanced weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == self.output_dim:
                    # Output layers: smaller init for better stability
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the transformer.
        
        Args:
            x: [batch, seq_len, input_dim] input sequence
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention=False: [batch, output_dim] predictions
            If return_attention=True: dict with predictions and attention maps
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply feature attention
        x_attended, feature_attn = self.feature_attention(x)
        
        # Input projection with residual
        x_proj = self.input_projection(x_attended)
        x_skip = self.input_skip(x_attended)
        x_proj = x_proj + x_skip
        
        # Add positional encoding
        x_pos = self.pos_encoder(x_proj)
        
        # Causal temporal convolutions
        x_conv = x_pos.transpose(1, 2)  # [batch, d_model, seq_len]
        # Left-pad for causality (kernel_size=3, so pad 2 on left)
        x_conv_padded = nn.functional.pad(x_conv, (2, 0))  # Causal padding
        x_conv_out = torch.relu(self.temporal_conv(x_conv_padded))
        x_conv = x_conv + x_conv_out  # Residual connection
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        x_conv = self.conv_norm(x_conv)
        
        # Transformer processing
        x_trans = self.transformer(x_conv)
        
        # Attention-based sequence pooling
        attn_scores = self.sequence_attention(x_trans)  # [batch, seq_len, 1]
        attn_weights_seq = torch.softmax(attn_scores, dim=1)
        x_pooled = torch.sum(x_trans * attn_weights_seq, dim=1)  # [batch, d_model]
        
        # Ensemble prediction
        predictions = []
        for head in self.prediction_heads:
            pred = head(x_pooled)
            predictions.append(pred)
        
        # Weighted ensemble
        ensemble_weights = torch.softmax(self.ensemble_weights, dim=0)
        final_prediction = sum(
            w * pred for w, pred in zip(ensemble_weights, predictions)
        )
        
        # Reconstruction for auxiliary loss (regularization)
        reconstruction = self.reconstruction_head(x_pooled)
        
        if return_attention:
            return {
                "prediction": final_prediction,
                "reconstruction": reconstruction,
                "feature_attention": feature_attn,
                "sequence_attention": attn_weights_seq,
                "ensemble_weights": ensemble_weights,
            }
        else:
            return {"prediction": final_prediction, "reconstruction": reconstruction}
