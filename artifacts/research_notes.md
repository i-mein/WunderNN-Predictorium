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

## Updated Results (Feb 2026)

### Best Results So Far
| Exp | Val Score | LB Score | Notes |
|-----|-----------|----------|-------|
| exp_20 | 0.2746 | **0.2858** | Ensemble (GRU 55% + Hybrid 45%) - BEST |
| exp_01 | 0.2310 | 0.2820 | Simple GRU, raw features |
| exp_02 | 0.2691 | 0.2766 | GRU + 49 engineered features |
| exp_13 | 0.2115 | 0.2262 | Hybrid Transformer-GRU |

### Failed Approaches
- exp_22 (2D Reshape Conv): 0.0266 val - 2D spatial structure doesn't work
- exp_21 (Multi-Stream Attention): 0.2009 val - worse than ensemble
- exp_03-05, exp_10: Leakage issues

### Gap to Leader: 0.324 - 0.2858 = 0.0382 (~4 percentage points)

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

## Next Experiments to Try (Priority Order)

### HIGH PRIORITY - Quick Wins

#### exp_23: Enhanced Feature Engineering for GRU
Based on DeepLOB and LOB research:
- **Micro-price**: `p_micro = (p_ask * v_bid + p_bid * v_ask) / (v_bid + v_ask)`
- **Order Flow Imbalance (OFI)**: `Δv_bid[0] - Δv_ask[0]` 
- **Weighted price pressure**: Sum of volume-weighted imbalances across levels
- **Price levels correlation**: Rolling corr between bid/ask at different levels
- Apply to exp_01 base (simple GRU) architecture

#### exp_24: Better Ensemble Weights
- Grid search exp_01 + exp_13 weights (test 60/40, 50/50, 70/30)
- Consider 3-model ensemble: exp_01 + exp_02 + exp_13
- Try stacking: train logistic regression on validation predictions

#### exp_25: Direct Pearson Loss
- Train exp_01 with Pearson correlation loss instead of MSE
- Pearson loss: `1 - correlation(pred, target)`
- May converge slower but directly optimizes competition metric

### MEDIUM PRIORITY - Architecture Improvements

#### exp_26: Temporal Convolutional Network (TCN) + GRU
- Dilated causal convolutions: receptive field grows exponentially
- Structure: `Input → DilatedConv(d=1,2,4,8) → GRU → Output`
- **MUST use left (causal) padding only**

#### exp_27: Layer Norm GRU + Residual
- Replace standard GRU with LayerNorm GRU
- Add residual connections between layers
- More stable training, potentially better generalization

#### exp_28: Separate t0/t1 Models
- Since t1 << t0 in performance, train specialized models
- Model A: Optimized for t0 (2/3 weight)
- Model B: Optimized for t1 (1/3 weight)
- Ensemble at inference

### LOWER PRIORITY - Research Ideas

#### exp_29: Temporal Fusion Transformer (Simplified)
- Variable selection network for feature importance
- Multi-head attention with positional encoding
- Interpretable attention weights

#### exp_30: Data Augmentation
- Add small noise to prices/volumes during training
- Mixup between similar sequences
- Time warping (stretch/compress sequences)

---

## Key Insights from Literature

### DeepLOB Architecture (Zhang et al., 2019)
- CNN layers capture spatial structure of LOB levels
- LSTM captures temporal dependencies
- Achieves 80%+ accuracy on FI-2010 (direction classification)
- Our task is regression, but patterns may transfer

### Autoformer (Wu et al., 2021)
- Series decomposition into trend + seasonal
- Auto-correlation mechanism better than attention for periodic data
- 38% improvement on long-term forecasting benchmarks
- Could help with t1 (longer horizon) prediction

### Optiver Competition Insights
- GBDTs (LightGBM, CatBoost) often outperform neural nets on tabular LOB data
- Cross-sectional features (compare current stock to market) are valuable
- Volume imbalance features consistently important

### t1 vs t0 Analysis
- t1 consistently ~30-35% of t0 correlation
- Weighted metric: 2/3 * t0 + 1/3 * t1
- **Focus on t0 first** - improving t0 by 0.03 ≈ improving t1 by 0.06

