import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (using Sinusoidal from class above instead of parameter for better generalization)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Pre-norm often stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model) # Final norm
        self.dropout_head = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, mask=None):
        # x: (Batch, Seq, Feature)
        x = self.input_proj(x)
        x = self.pos_encoder(x) # (Batch, Seq, d_model) with auto transpose handle
        
        seq_len = x.size(1)
        
        if mask is None:
             # Causal mask: Ensure query i cannot attend to key j > i
             mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer Encoder
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        output = self.norm(output)
        output = self.dropout_head(output)
        
        predictions = self.head(output)
        return predictions, None
