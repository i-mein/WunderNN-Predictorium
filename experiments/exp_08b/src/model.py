"""
Model architecture for exp_09: Dual-Head GRU

Simple but effective architecture:
- Shared GRU backbone (like exp_06)
- Separate output heads for t0 and t1
- No complex CNN/attention (fast inference)
"""
import torch
import torch.nn as nn


class DualHeadGRU(nn.Module):
    """
    Dual-head unidirectional GRU for exp_09.
    
    Architecture:
    - Shared GRU backbone (256 hidden, 2 layers)
    - Separate FC heads for t0 and t1
    
    Key design:
    - Unidirectional only (no future leakage)
    - Simple architecture for fast inference
    - Separate heads allow task-specific optimization
    """
    
    def __init__(self, input_dim=65, hidden_dim=256, num_layers=2,
                 head_hidden=128, dropout=0.3, dropout_t0=0.2, dropout_t1=0.3):
        super(DualHeadGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared GRU backbone
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: No future peeking
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # t0 head (stronger predictor - lower dropout)
        self.t0_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_t0),
            nn.Linear(head_hidden, 1)
        )
        
        # t1 head (weaker predictor - higher dropout)
        self.t1_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_t1),
            nn.Linear(head_hidden, 1)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Hidden state from previous step (optional)
        
        Returns:
            pred_t0: Predictions for t0 (batch, seq_len, 1)
            pred_t1: Predictions for t1 (batch, seq_len, 1)
            hidden: Updated hidden state
        """
        # Shared backbone
        out, hidden = self.gru(x, hidden)
        
        # Separate heads
        pred_t0 = self.t0_head(out)
        pred_t1 = self.t1_head(out)
        
        return pred_t0, pred_t1, hidden
    
    def forward_combined(self, x, hidden=None):
        """
        Forward pass returning combined output (for compatibility with exp_06 eval).
        
        Returns:
            predictions: (batch, seq_len, 2) with [t0, t1] concatenated
            hidden: Updated hidden state
        """
        pred_t0, pred_t1, hidden = self.forward(x, hidden)
        predictions = torch.cat([pred_t0, pred_t1], dim=-1)
        return predictions, hidden
