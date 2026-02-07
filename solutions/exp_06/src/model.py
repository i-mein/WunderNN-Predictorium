import torch
import torch.nn as nn

class UnidirectionalGRU(nn.Module):
    """
    Leakage-free unidirectional GRU for exp_06.
    
    Key design principles:
    - Unidirectional only (no bidirectional to avoid future peeking)
    - Stateful inference matches training exactly
    - Simple architecture proven in exp_01/02
    """
    
    def __init__(self, input_dim=65, hidden_dim=256, num_layers=2, 
                 output_dim=2, dropout=0.3):
        super(UnidirectionalGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Unidirectional GRU (CRITICAL: bidirectional=False)
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: No future peeking
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Hidden state from previous step (optional)
        
        Returns:
            predictions: Output tensor of shape (batch, seq_len, output_dim)
            hidden: Updated hidden state
        """
        out, hidden = self.gru(x, hidden)
        predictions = self.fc(out)
        return predictions, hidden
