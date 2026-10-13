import torch
import torch.nn as nn

class CNNGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3):
        super(CNNGRUModel, self).__init__()
        
        # 1D Convolution
        # Input shape to Conv1d: (Batch, Channels/Feats, Seq_Len)
        # Note: GRU expects (Batch, Seq, Feats) if batch_first=True
        # We need to transpose before Conv and after Conv.
        
        # We want to convolve over time? Or over features?
        # Usually for Time Series, Conv1d over Time dimension extracts local temporal patterns.
        # Input channels = Features (input_dim).
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2 # Same padding roughly
        )
        self.relu = nn.ReLU()
        
        # GRU
        # Input to GRU: cnn_channels
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (Batch, Seq, Feats) = B x T x F
        
        # Prepare for Conv1d: (Batch, Feats, Seq) = B x F x T
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        
        # Back to (Batch, Seq, Channels) for GRU = B x T x F
        x = x.permute(0, 2, 1)
        
        # GRU
        out, hidden = self.gru(x, hidden)
        
        # Head (on all steps)
        # We want predictions for all steps for training
        outputs = self.fc(out)
        
        return outputs, hidden
