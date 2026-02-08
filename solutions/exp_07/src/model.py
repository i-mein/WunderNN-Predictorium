import torch
import torch.nn as nn
import torch.nn.functional as F

class RawFeatureStream(nn.Module):
    """
    1D-CNN stream for processing raw LOB features.
    Captures local patterns in high-frequency market data.
    """
    
    def __init__(self, input_dim=32, channels=[64, 128, 128], kernels=[3, 5, 7], dropout=0.2):
        super(RawFeatureStream, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_channels = input_dim
        self.kernels = kernels
        for out_channels, kernel_size in zip(channels, kernels):
            # Causal CNN: No padding in Conv1d, manual left padding in forward
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=0))
            self.bns.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = channels[-1]
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        # Transpose for Conv1D: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        for conv, bn, k in zip(self.convs, self.bns, self.kernels):
            # Causal padding: pad left by (k - 1)
            padding = k - 1
            if padding > 0:
                x = F.pad(x, (padding, 0))
            
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Transpose back: (batch, seq_len, output_dim)
        x = x.transpose(1, 2)
        return x


class EngFeatureStream(nn.Module):
    """
    GRU stream for processing engineered features.
    Models temporal dependencies in domain-informed features.
    """
    
    def __init__(self, input_dim=43, hidden_dim=256, num_layers=2, dropout=0.3):
        super(EngFeatureStream, self).__init__()
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,  # Unidirectional for leakage-free
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_dim = hidden_dim
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            hidden: Optional hidden state
        Returns:
            output: (batch, seq_len, hidden_dim)
            hidden: Updated hidden state
        """
        output, hidden = self.gru(x, hidden)
        return output, hidden


class FeatureFusion(nn.Module):
    """
    Attention-based fusion of raw and engineered feature streams.
    """
    
    def __init__(self, fusion_dim=384, num_heads=8, dropout=0.1):
        super(FeatureFusion, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, fusion_dim) - concatenated streams
        Returns:
            (batch, seq_len, fusion_dim) - fused representation
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection + layer norm
        x = self.layer_norm(x + self.dropout(attn_output))
        return x


class TaskHead(nn.Module):
    """
    Task-specific head for t0 or t1 prediction.
    """
    
    def __init__(self, input_dim=384, hidden_dim=128, num_heads=4, dropout=0.2):
        super(TaskHead, self).__init__()
        
        # Task-specific GRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            hidden: Optional hidden state
        Returns:
            predictions: (batch, seq_len, 1)
            hidden: Updated hidden state
        """
        # GRU processing
        gru_out, hidden = self.gru(x, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Residual connection
        out = gru_out + attn_out
        
        # Output projection
        predictions = self.fc(out)
        return predictions, hidden


class DualHeadDualStream(nn.Module):
    """
    Main model: Dual-head dual-stream architecture for exp_07.
    
    Architecture:
    - Raw features → 1D-CNN stream
    - Engineered features → GRU stream
    - Fusion → Attention-based combination
    - Dual heads → Separate t0 and t1 predictions
    """
    
    def __init__(self, raw_dim=32, eng_dim=43, 
                 cnn_channels=[64, 128, 128], cnn_kernels=[3, 5, 7],
                 gru_hidden=256, gru_layers=2,
                 fusion_heads=8, head_hidden=128, head_attention_heads=4,
                 dropout_t0=0.2, dropout_t1=0.3):
        super(DualHeadDualStream, self).__init__()
        
        # Feature streams
        self.raw_stream = RawFeatureStream(
            input_dim=raw_dim,
            channels=cnn_channels,
            kernels=cnn_kernels,
            dropout=dropout_t0
        )
        
        self.eng_stream = EngFeatureStream(
            input_dim=eng_dim,
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout_t1
        )
        
        # Fusion
        fusion_dim = self.raw_stream.output_dim + self.eng_stream.output_dim
        self.fusion = FeatureFusion(
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=0.1
        )
        
        # Task heads
        self.t0_head = TaskHead(
            input_dim=fusion_dim,
            hidden_dim=head_hidden,
            num_heads=head_attention_heads,
            dropout=dropout_t0
        )
        
        self.t1_head = TaskHead(
            input_dim=fusion_dim,
            hidden_dim=head_hidden,
            num_heads=head_attention_heads,
            dropout=dropout_t1
        )
        
        self.fusion_dim = fusion_dim
    
    def forward(self, raw_features, eng_features, 
                eng_hidden=None, t0_hidden=None, t1_hidden=None):
        """
        Forward pass.
        
        Args:
            raw_features: (batch, seq_len, raw_dim)
            eng_features: (batch, seq_len, eng_dim)
            eng_hidden: Hidden state for eng_stream GRU
            t0_hidden: Hidden state for t0_head GRU
            t1_hidden: Hidden state for t1_head GRU
        
        Returns:
            pred_t0: (batch, seq_len, 1)
            pred_t1: (batch, seq_len, 1)
            eng_hidden: Updated eng_stream hidden state
            t0_hidden: Updated t0_head hidden state
            t1_hidden: Updated t1_head hidden state
        """
        # Process streams
        raw_out = self.raw_stream(raw_features)
        eng_out, eng_hidden = self.eng_stream(eng_features, eng_hidden)
        
        # Concatenate streams
        fused = torch.cat([raw_out, eng_out], dim=-1)
        
        # Fusion with attention
        fused = self.fusion(fused)
        
        # Task-specific predictions
        pred_t0, t0_hidden = self.t0_head(fused, t0_hidden)
        pred_t1, t1_hidden = self.t1_head(fused, t1_hidden)
        
        return pred_t0, pred_t1, eng_hidden, t0_hidden, t1_hidden
