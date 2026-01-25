import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.3, attention_dim=64):
        super(AttentionGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        self.gru = nn.GRU(cnn_channels, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        # Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (Batch, Seq, Feats)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1) 
        
        # GRU
        gru_out, hidden = self.gru(x, hidden) # (Batch, Seq, Hidden)
        
        # Causal Attention
        # Create mask: (Seq, Seq). Upper triangular is -inf.
        seq_len = gru_out.size(1)
        # mask shape: (N*num_heads, L, S)? No, (L, S) or (N, L, S).
        # nn.MultiheadAttention takes attn_mask (L, S) or (N*heads, L, S).
        # If batch_first=True, mask matches Query sequence logic.
        
        # Causal mask: position i can attend to 0..i.
        # So mask[i, j] = 0 if j <= i else -inf.
        # This is strictly lower triangular.
        # Usually torch.triu(ones) == 1 is masked (future).
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # attn output: (Batch, Seq, Hidden)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out, attn_mask=mask, is_causal=True)
        
        # Residual connection?
        out = gru_out + attn_out
        
        outputs = self.fc(out)
        return outputs, hidden
