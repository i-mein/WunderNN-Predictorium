import torch
import torch.nn as nn

class CNNGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (Batch, Seq, Feats) -> (Batch, Feats, Seq)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1) # -> (Batch, Seq, Channels)
        out, hidden = self.gru(x, hidden)
        outputs = self.fc(out)
        return outputs, hidden
