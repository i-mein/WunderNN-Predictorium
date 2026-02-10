# Hybrid CNN-GRU-Transformer Architecture Research

## Research from Kaggle Solutions and Academic Papers

### 1. DeepLOB (Zhang et al., 2019) - arXiv:1808.03668

**"Deep Convolutional Neural Networks for Limit Order Books"**

This is the seminal paper for LOB prediction with deep learning:

**Architecture**:
```
Input LOB Data (batch, time, 40 features)
        ↓
[Inception Module]  ← Multi-scale 1D Conv (k=1,2,3)
        ↓
[Inception Module]  ← Second inception layer
        ↓
[LSTM Layer]        ← Sequential dependencies (64 hidden)
        ↓
[Dense Layers]      ← FC → Softmax
        ↓
Output (Up/Down/Stationary classification)
```

**Key Insights**:
1. **Inception-like CNN**: Uses multiple kernel sizes (1, 2, 3) in parallel to capture different temporal scales
2. **CNN before RNN**: Feature extraction first, then temporal modeling
3. **Relatively shallow**: Only 2 inception modules + 1 LSTM layer
4. **Works on raw LOB data**: No heavy feature engineering needed

**Performance**: State-of-the-art on FI-2010 benchmark (72% accuracy on mid-price movement)

---

### 2. Transformers for LOB (Wallbridge, 2020) - arXiv:2003.00130

**"Transformers for Limit Order Books"**

Introduced transformers to LOB prediction:

**Architecture**:
```
Input LOB Data (batch, time, features)
        ↓
[Causal Conv Stack]  ← Dilated causal convolutions (no future leakage)
        ↓
[Positional Encoding]
        ↓
[Transformer Encoder] ← Masked self-attention (4 heads, 2 layers)
        ↓
[Output Head]
        ↓
Predictions
```

**Key Insights**:
1. **Causal convolutions**: Ensures no information from future timesteps leaks
2. **Masked self-attention**: Only attends to past positions
3. **Outperforms DeepLOB** on FI-2010 benchmark
4. **Attention visualization** shows model focuses on recent orderbook states

---

### 3. Kaggle Competition Solutions

#### Optiver Realized Volatility (2021) - $100K prize
- **1st place**: Ensemble of LightGBM + Neural Networks
- **Top DL approaches**: CNN feature extractors + GRU/LSTM
- **Key insight**: Feature engineering was more important than architecture

#### Ubiquant Market Prediction (2022) - $100K prize  
- **Metric**: Mean Pearson correlation (similar to our task!)
- **1st place**: MLP with careful feature engineering
- **2nd/3rd place**: GRU + Attention mechanisms
- **Key insight**: Simple models with good features beat complex architectures

#### Jane Street Market Prediction (2021) - $100K prize
- **1st place**: Ensemble of MLPs with careful regularization
- **Key insight**: Overfitting was the main challenge (like our exp_09)

#### G-Research Crypto Forecasting (2022) - $125K prize
- **Metric**: Weighted correlation (similar to ours)
- **Top solutions**: LightGBM + Neural network ensembles
- **Key insight**: Temporal validation crucial for avoiding leakage

---

## Proposed Hybrid Architectures for exp_10

Based on the research, here are 3 architecture options ranked by complexity:

### Option A: DeepLOB-Style (CNN + LSTM)

Closest to the proven DeepLOB architecture:

```python
class DeepLOBStyle(nn.Module):
    """
    DeepLOB-inspired architecture for our LOB prediction task.
    
    Changes from original:
    - Use GRU instead of LSTM (faster, similar performance)
    - Dual heads for t0/t1 (regression instead of classification)
    - Heavier regularization (dropout, weight decay)
    """
    
    def __init__(self, input_dim=32, dropout=0.4):
        super().__init__()
        
        # Inception Module 1
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            ),
        ])
        
        # Inception Module 2
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(48, 16, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            ),
            nn.Sequential(
                nn.Conv1d(48, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            ),
        ])
        
        self.dropout_cnn = nn.Dropout(dropout * 0.5)
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            32,  # 16 + 16 from inception2
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )
        
        # Dual heads
        self.t0_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, 1)
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, hidden=None):
        # x: (batch, seq, features)
        batch, seq_len, _ = x.shape
        
        # Transpose for Conv1d: (batch, features, seq)
        x = x.transpose(1, 2)
        
        # Inception 1
        inception1_out = torch.cat([conv(x) for conv in self.inception1], dim=1)
        inception1_out = self.dropout_cnn(inception1_out)
        
        # Inception 2
        inception2_out = torch.cat([conv(inception1_out) for conv in self.inception2], dim=1)
        inception2_out = self.dropout_cnn(inception2_out)
        
        # Transpose back: (batch, seq, channels)
        x = inception2_out.transpose(1, 2)
        
        # GRU
        gru_out, hidden = self.gru(x, hidden)
        
        # Dual heads
        pred_t0 = self.t0_head(gru_out)
        pred_t1 = self.t1_head(gru_out)
        
        return pred_t0, pred_t1, hidden
```

**Pros**: Proven architecture, multi-scale features, reasonable complexity
**Cons**: More parameters than simple GRU, may overfit

---

### Option B: Transformer-LOB Style (Causal Conv + Attention)

Closer to the Transformer-LOB paper:

```python
class TransformerLOBStyle(nn.Module):
    """
    Transformer-inspired architecture with causal constraints.
    
    Key features:
    - Causal dilated convolutions (no future leakage)
    - Lightweight self-attention (2 heads)
    - Pre-LayerNorm for training stability
    """
    
    def __init__(self, input_dim=32, d_model=128, nhead=2, num_layers=2, dropout=0.4):
        super().__init__()
        
        # Causal convolutional stack
        self.causal_conv = nn.Sequential(
            # Conv with dilation=1
            nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            
            # Conv with dilation=2 (larger receptive field)
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        
        # Make convs causal by chopping off future values
        self.chop_future = True
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.t0_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, 1)
        )
        
        self.t1_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # Causal conv: (batch, features, seq)
        x_conv = x.transpose(1, 2)
        x_conv = self.causal_conv(x_conv)
        
        # Chop future: remove extra padding from causal conv
        x_conv = x_conv[:, :, :seq_len]
        
        # Back to (batch, seq, d_model)
        x = x_conv.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Causal mask for transformer
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), 
            diagonal=1
        ).bool()
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)
        
        # Output
        pred_t0 = self.t0_head(x)
        pred_t1 = self.t1_head(x)
        
        return pred_t0, pred_t1
```

**Pros**: Attention mechanism for long-range dependencies, proven in LOB research
**Cons**: More complex, attention is O(n²), may be slow for inference

---

### Option C: Lightweight Hybrid (Recommended for exp_10)

Best balance of proven techniques with minimal complexity:

```python
class LightweightHybrid(nn.Module):
    """
    Lightweight hybrid combining best elements:
    - Simple CNN for local patterns (like DeepLOB)
    - GRU for temporal dynamics (proven in exp_01/06)
    - Optional lightweight attention for t1 (harder target)
    
    Key design choices:
    - Small model (reduce overfitting)
    - Heavy regularization
    - Raw features only (32 features, like exp_01)
    """
    
    def __init__(
        self,
        input_dim=32,
        cnn_filters=32,
        hidden_dim=192,
        num_layers=2,
        use_attention_t1=True,
        attention_heads=2,
        dropout=0.4,
    ):
        super().__init__()
        
        self.use_attention_t1 = use_attention_t1
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Simple CNN block
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
            nn.Dropout(dropout * 0.5),
        )
        
        # GRU backbone
        self.gru = nn.GRU(
            cnn_filters,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: No future leakage
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Optional attention for t1
        if use_attention_t1:
            self.t1_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.t1_attn_norm = nn.LayerNorm(hidden_dim)
        
        # t0 head (simpler, direct from GRU)
        self.t0_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, 1)
        )
        
        # t1 head (with attention context)
        self.t1_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, hidden=None):
        batch, seq_len, features = x.shape
        
        # Normalize input
        x = self.input_norm(x)
        
        # CNN: (batch, seq, feat) → (batch, feat, seq) → CNN → (batch, cnn_filters, seq)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq, cnn_filters)
        
        # GRU
        gru_out, hidden = self.gru(x_cnn, hidden)  # (batch, seq, hidden_dim)
        
        # t0: Direct from GRU
        pred_t0 = self.t0_head(gru_out)
        
        # t1: With attention
        if self.use_attention_t1:
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), 
                diagonal=1
            ).bool()
            
            # Self-attention
            attn_out, _ = self.t1_attention(
                gru_out, gru_out, gru_out, 
                attn_mask=causal_mask
            )
            
            # Residual + LayerNorm
            t1_features = self.t1_attn_norm(gru_out + attn_out)
            pred_t1 = self.t1_head(t1_features)
        else:
            pred_t1 = self.t1_head(gru_out)
        
        return pred_t0, pred_t1, hidden
```

**Pros**: 
- Combines proven techniques (CNN feature extraction, GRU temporal, attention for hard target)
- Small and fast (meets inference constraints)
- Heavy regularization to prevent overfitting
- Uses raw 32 features (like best-performing exp_01)

**Cons**: May not capture as complex patterns as full DeepLOB/Transformer

---

## Recommended Approach for exp_10

### Primary: Option C (Lightweight Hybrid)

Given our constraints:
1. **Inference time limit**: Need fast model
2. **Overfitting tendency**: exp_09 overfit badly
3. **Best LB was exp_01**: Simple architecture with raw features

The lightweight hybrid is the best balance.

### Training Strategy

```python
# Model config
config = {
    'input_dim': 32,           # Raw features only!
    'cnn_filters': 32,
    'hidden_dim': 192,         # Smaller than exp_06 (256)
    'num_layers': 2,
    'use_attention_t1': True,  # Help with harder target
    'attention_heads': 2,
    'dropout': 0.4,            # Higher than exp_06 (0.3)
}

# Training config
training = {
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,      # Strong regularization
    'epochs': 25,              # Short training
    'gradient_clip': 0.5,
    'early_stopping_patience': 8,
    
    # Loss
    'loss': 'huber',           # More robust than MSE
    'huber_delta': 1.0,
    't0_weight': 0.6,
    't1_weight': 0.4,
    
    # Augmentation
    'input_noise_std': 0.01,
    'feature_dropout': 0.05,
}
```

### Alternative: exp_10b with pure DeepLOB

If Option C doesn't improve, try Option A (DeepLOB-style) with same regularization settings.

---

## Implementation Priority

1. ✅ Research complete (this document)
2. [ ] Implement `LightweightHybrid` model
3. [ ] Create training script with augmentation
4. [ ] Train with WandB monitoring
5. [ ] Compare val scores with exp_06/exp_09
6. [ ] If promising, create submission

---

## References

1. Zhang et al. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books". arXiv:1808.03668
2. Wallbridge (2020). "Transformers for Limit Order Books". arXiv:2003.00130
3. Kaggle Optiver Realized Volatility 2021 - Competition discussions
4. Kaggle Ubiquant Market Prediction 2022 - Competition discussions
5. Kaggle Jane Street Market Prediction 2021 - Competition discussions
