# Research Notes: LOB Prediction Architectures

## Key Insights from Similar Competitions

### DeepLOB (Zhang et al., IEEE 2019)
- **Architecture**: Conv2D → Inception → LSTM
- **Key Innovation**: Transform LOB data to 2D image (T × Features × 1)
- **Input**: 100 timesteps × 40 features (10 levels × 4 values)
- **Conv Blocks**: 
  - Block 1: (1,2) stride reduces feature dim
  - Block 2-3: (4,1) temporal convolutions with same padding
  - Final: Reduce to 1 feature dimension
- **Inception Module**: Multi-scale temporal patterns (1x1, 3x1, 5x1 + MaxPool)
- **Output**: LSTM → Dense for classification
- **Total params**: ~143K

### FI-2010 Dataset Characteristics
- 5 stocks, 10 days of data
- LOB updates every ~10ms
- Task: Predict price direction (up/down/stay)
- 10 levels of bid/ask prices and volumes

### Optiver Trading at the Close Competition
- **Winning approaches**: LightGBM/CatBoost ensembles dominated
- **Key features**: Cross-sectional features, weighted prices, volume imbalances
- **Neural nets**: Generally didn't outperform GBDTs for tabular LOB data

### Jane Street Market Prediction
- **Insight**: Feature engineering crucial for LOB data
- **Rolling/lagged features**: Dangerous if not properly causal
- **Cross-stock features**: Can capture market microstructure

## Architectural Patterns That Generalize

### 1. Simple GRU (Best LB: 0.2820 - exp_01)
- 2 layers, 256 hidden
- Raw 32 features only
- MSE loss
- **Why it works**: No feature leakage, captures temporal patterns

### 2. Hybrid Transformer-GRU (LB: 0.2262 - exp_13)
- Transformer for local attention
- GRU for temporal memory
- 7 engineered features (leakage-free)
- **Why it works**: Balanced complexity, proper attention patterns

### 3. What Doesn't Work
- Symmetric padding in convolutions (leakage)
- Rolling window features computed in model (leakage)
- Complex attention across full sequence (overfitting)
- Very deep networks (overfitting)

## Feature Engineering Best Practices

### Safe Features (No Look-Ahead)
1. **Mid-price**: (best_bid + best_ask) / 2
2. **Spread**: best_ask - best_bid
3. **Volume imbalance**: (bid_vol - ask_vol) / (bid_vol + ask_vol)
4. **Level imbalance**: v0 - v6 (best level volumes)
5. **Weighted mid**: Volume-weighted average price
6. **Price momentum**: Sum of delta prices (dp0-dp3)
7. **Volume momentum**: Sum of delta volumes (dv0-dv3)

### Temporal Differences
1. **1st order**: x[t] - x[t-1]
2. **2nd order**: diff1[t] - diff1[t-1] (acceleration)
3. **Cross-differences**: bid[t] - ask[t] at each level

### Dangerous Features (Leakage Risk)
- Rolling mean/std over window
- Future normalization
- Cross-sequence statistics
- Non-causal convolutions

## Architecture Recommendations

### For Generalization
1. **Model size**: 20K-200K parameters optimal
2. **Regularization**: Dropout 0.15-0.25
3. **Loss**: MSE more stable than Pearson for training
4. **Inference**: Single-step with hidden state carryover

### For Speed
1. **Avoid**: Per-step pandas operations
2. **Use**: NumPy for feature computation
3. **Batch**: Process full sequences during training
4. **Single-step**: Carry hidden state during inference

## Next Experiments to Try

### exp_20: Simpler Conv1D
- 1D causal convolutions on raw features
- Dilated convolutions for multi-scale
- GRU for final temporal modeling

### exp_21: Attention-Only
- Local attention (window = 50)
- Relative positional encoding
- No RNN component

### exp_22: Feature Selection
- Use exp_13 features with exp_01 architecture
- Focus on most predictive features
- L1 regularization for sparsity

### exp_23: Ensemble
- Combine exp_01, exp_13, best Conv2D
- Simple averaging or learned weights
- Diversity through different architectures
