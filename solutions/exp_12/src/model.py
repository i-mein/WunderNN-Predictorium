"""
Model for exp_12: Specialized t1 Prediction

Key insight: t0 (~0.37 Pearson) is easier than t1 (~0.14 Pearson).
This model has asymmetric heads:
- Simple head for t0 (already works well)
- Deeper head for t1 (needs more capacity)
"""

import torch
import torch.nn as nn


class AsymmetricDualHeadGRU(nn.Module):
    """
    GRU with asymmetric dual heads for t0/t1 prediction.
    
    Architecture:
    - Shared GRU backbone
    - Simple t0 head (1 hidden layer)
    - Deep t1 head (2 hidden layers, more capacity)
    """
    
    def __init__(
        self,
        input_dim=32,
        gru_hidden=256,
        gru_layers=2,
        gru_dropout=0.3,
        t0_hidden=64,
        t0_dropout=0.2,
        t1_hidden1=128,
        t1_hidden2=64,
        t1_dropout=0.4,
    ):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Shared GRU backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout if gru_layers > 1 else 0.0
        )
        
        # Layer norm after GRU
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        # Simple t0 head (1 hidden layer)
        self.t0_head = nn.Sequential(
            nn.Linear(gru_hidden, t0_hidden),
            nn.GELU(),
            nn.Dropout(t0_dropout),
            nn.Linear(t0_hidden, 1)
        )
        
        # Deep t1 head (2 hidden layers for more capacity)
        self.t1_head = nn.Sequential(
            nn.Linear(gru_hidden, t1_hidden1),
            nn.GELU(),
            nn.Dropout(t1_dropout),
            nn.Linear(t1_hidden1, t1_hidden2),
            nn.GELU(),
            nn.Dropout(t1_dropout),
            nn.Linear(t1_hidden2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gru' in name.lower():
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            hidden: GRU hidden state
        
        Returns:
            pred_t0: (batch, seq_len, 1)
            pred_t1: (batch, seq_len, 1)
            hidden: Updated hidden state
        """
        # Input normalization
        x = self.input_norm(x)
        
        # GRU
        gru_out, hidden = self.gru(x, hidden)
        
        # Layer norm
        gru_out = self.gru_norm(gru_out)
        
        # Dual heads
        pred_t0 = self.t0_head(gru_out)
        pred_t1 = self.t1_head(gru_out)
        
        return pred_t0, pred_t1, hidden


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = AsymmetricDualHeadGRU()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 100, 32)
    pred_t0, pred_t1, hidden = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output t0 shape: {pred_t0.shape}")
    print(f"Output t1 shape: {pred_t1.shape}")
    print(f"Hidden shape: {hidden.shape}")
