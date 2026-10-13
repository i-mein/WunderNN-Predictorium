import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer GRU
        # batch_first=True expects input of shape (batch, seq, feature)
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        # hidden: (num_layers, batch, hidden_dim)
        
        out, hidden = self.gru(x, hidden)
        
        # out: (batch, seq_len, hidden_dim)
        
        # We process the entire sequence and apply FC to each step
        # This is useful for training on full sequences
        predictions = self.fc(out)
        
        # predictions: (batch, seq_len, output_dim)
        return predictions, hidden
