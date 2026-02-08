# Implementation Plan: Experiment 07 - Advanced Dual-Head Architecture

## Goal

Build an advanced model for exp_07 that improves upon exp_06's 0.2577 validation score by incorporating:
1. **Dual-head architecture** with separate outputs for t0 and t1
2. **Dual-stream feature processing** (raw features vs engineered features)
3. **Task-specific loss functions** optimized for each target's characteristics
4. **SOTA techniques** from Optiver and financial time series competitions
5. **Leakage-free design** (maintaining exp_06's principles)

## Background Analysis

### Exp_06 Results Analysis

| Metric | Value | Observation |
|:---|:---|:---|
| Best Val Score | 0.2577 | Between exp_01 (0.2310) and exp_02 (0.2691) |
| **t0 Correlation** | **0.3707** | Strong! Better than exp_01 (0.3342) |
| **t1 Correlation** | **0.1447** | Weaker, but better than exp_01 (0.1277) |
| Gap (t0 - t1) | 0.2260 | **Large disparity** suggests different dynamics |

> [!IMPORTANT]
> **Key Insight**: The large gap between t0 (0.3707) and t1 (0.1447) correlations suggests that:
> 1. t0 and t1 have **different predictive patterns**
> 2. A single-head model with uniform loss may be **suboptimal**
> 3. **Separate heads** with task-specific architectures could improve performance

### SOTA Research Findings

From Optiver Realized Volatility Prediction (1st place solution):
- **Architecture**: LightGBM + 1D-CNN + MLP ensemble
- **Key Technique**: NearestNeighbor aggregation features (360 features from similar time-ids/stocks)
- **Feature Engineering**: Dominates over complex architectures
- **Insight**: Hybrid models (GBDT + DL) work well for financial time series

From Financial Time Series Competitions (5th place solution):
- **Architecture**: 6-layer Transformer with **optimized Pearson Correlation loss**
- **Data Augmentation**: Random zeroing, masking
- **Ensemble**: 5 models for robustness
- **Insight**: Direct optimization of Pearson correlation can be effective

### Feature Importance Analysis (from exp_06)

**Raw Features** (32):
- Direct market data: prices (p0-p11), volumes (v0-v11), deltas (dp0-dp3, dv0-dv3)
- Characteristics: High-frequency, noisy, but contains raw market microstructure

**Engineered Features** (33):
- Mid-prices, spreads, volume imbalances, ratios, rolling stats, diffs
- Characteristics: Smoothed, domain-informed, captures market dynamics

> [!NOTE]
> **Hypothesis**: Raw and engineered features capture **different aspects** of market behavior. Processing them in **separate streams** before fusion could improve representation learning.

## Proposed Architecture: Dual-Head Dual-Stream Model

### High-Level Design

```
Input (65 features)
    ├─ Raw Features (32) ────────┐
    │                            │
    │  ┌─ 1D-CNN (3 layers)      │
    │  │  - Kernel sizes: 3,5,7  │
    │  │  - Channels: 64,128,128 │
    │  └─ Output: (seq, 128)     │
    │                            │
    └─ Engineered Features (33) ─┤
                                 │
       ┌─ GRU (2 layers)         │
       │  - Hidden: 256          │
       │  - Unidirectional       │
       └─ Output: (seq, 256)     │
                                 │
                    ↓            │
            Feature Fusion       │
         (Concat + Attention)    │
                    ↓            │
         Shared Representation   │
              (seq, 384)         │
                    ↓            │
         ┌──────────┴──────────┐
         │                     │
    t0 Head (Strong)      t1 Head (Weak)
    ├─ GRU (128)          ├─ GRU (128)
    ├─ Attention          ├─ Attention
    ├─ FC (128→64→1)      ├─ FC (128→64→1)
    └─ Output: t0         └─ Output: t1
```

### Component Details

#### 1. Dual-Stream Feature Processing

**Raw Feature Stream** (1D-CNN):
- **Purpose**: Capture local patterns in high-frequency market data
- **Architecture**:
  - Conv1D(32 → 64, kernel=3) + ReLU + BatchNorm
  - Conv1D(64 → 128, kernel=5) + ReLU + BatchNorm
  - Conv1D(128 → 128, kernel=7) + ReLU + BatchNorm
  - Residual connections between layers
- **Output**: (batch, seq_len, 128)

**Engineered Feature Stream** (GRU):
- **Purpose**: Model temporal dependencies in domain-informed features
- **Architecture**:
  - 2-layer unidirectional GRU
  - Hidden dim: 256
  - Dropout: 0.3
- **Output**: (batch, seq_len, 256)

#### 2. Feature Fusion Module

**Fusion Strategy**: Attention-based fusion
```python
# Concatenate streams
fused = concat([raw_stream, eng_stream], dim=-1)  # (batch, seq, 384)

# Self-attention for feature interaction
attn_output = MultiHeadAttention(
    embed_dim=384,
    num_heads=8,
    dropout=0.1
)(fused, fused, fused)

# Residual connection
shared_repr = LayerNorm(fused + attn_output)
```

#### 3. Dual-Head Architecture

**t0 Head** (Strong predictor):
- **Architecture**:
  - GRU(384 → 128, 1 layer)
  - Self-attention (4 heads)
  - FC(128 → 64) + ReLU + Dropout(0.2)
  - FC(64 → 1)
- **Loss**: Weighted Pearson Correlation Loss (weight: 0.7)

**t1 Head** (Weak predictor):
- **Architecture**:
  - GRU(384 → 128, 1 layer)
  - Self-attention (4 heads)
  - FC(128 → 64) + ReLU + Dropout(0.3)  # Higher dropout
  - FC(64 → 1)
- **Loss**: Weighted Pearson Correlation Loss (weight: 0.3) + MSE regularization

> [!NOTE]
> **Design Rationale**: t1 is harder to predict (lower correlation), so we:
> 1. Use higher dropout (0.3 vs 0.2) to prevent overfitting
> 2. Add MSE regularization to stabilize training
> 3. Give it lower weight in total loss (0.3 vs 0.7)

### Loss Function Design

#### Pearson Correlation Loss

```python
def pearson_correlation_loss(y_pred, y_true, weights=None):
    """
    Negative Pearson correlation as loss (to maximize correlation).
    Optionally weighted by target amplitude.
    """
    if weights is None:
        weights = torch.abs(y_true)
    
    # Weighted means
    mean_pred = (y_pred * weights).sum() / weights.sum()
    mean_true = (y_true * weights).sum() / weights.sum()
    
    # Centered values
    pred_centered = y_pred - mean_pred
    true_centered = y_true - mean_true
    
    # Weighted covariance and variances
    cov = (weights * pred_centered * true_centered).sum() / weights.sum()
    var_pred = (weights * pred_centered ** 2).sum() / weights.sum()
    var_true = (weights * true_centered ** 2).sum() / weights.sum()
    
    # Pearson correlation
    corr = cov / (torch.sqrt(var_pred * var_true) + 1e-8)
    
    # Return negative (to minimize)
    return -corr
```

#### Combined Loss

```python
# t0 loss (strong predictor)
loss_t0 = pearson_correlation_loss(pred_t0, target_t0)

# t1 loss (weak predictor) - add MSE regularization
loss_t1_pearson = pearson_correlation_loss(pred_t1, target_t1)
loss_t1_mse = F.mse_loss(pred_t1, target_t1)
loss_t1 = loss_t1_pearson + 0.1 * loss_t1_mse  # 10% MSE regularization

# Combined loss
total_loss = 0.7 * loss_t0 + 0.3 * loss_t1
```

## Feature Engineering

### Maintain exp_06's Leakage-Free Features (65)

- ✅ All features from exp_06 (raw 32 + engineered 33)
- ✅ Proper `groupby('seq_ix')` for rolling/diff operations
- ✅ `min_periods=1` for all rolling windows
- ✅ No future information

### Additional Features (Optional, +10 features)

**Cross-Feature Interactions** (5):
1. `mid_price_0 * vol_imbal_total` (price-volume interaction)
2. `spread_0 / mid_price_0` (relative spread, already have this)
3. [(p0 - p1) / spread_0](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_08/solution.py#169-178) (depth ratio)
4. `vol_imbal_0 * vol_imbal_1` (level interaction)
5. `price_depth * vol_ratio` (depth-volume interaction)

**Advanced Rolling Stats** (5):
1. `mid_price_0_roll_skew_10` (asymmetry)
2. `spread_0_roll_kurt_10` (tail behavior)
3. `vol_imbal_total_roll_min_10`
4. `vol_imbal_total_roll_max_10`
5. `mid_price_0_roll_range_10` (max - min)

**Total Features**: 65 + 10 = **75 features**

## Training Configuration

### Model Hyperparameters

```python
# Feature streams
RAW_FEATURES = 32
ENG_FEATURES = 33 (or 43 with additional features)

# CNN stream
CNN_CHANNELS = [64, 128, 128]
CNN_KERNELS = [3, 5, 7]

# GRU stream
GRU_HIDDEN = 256
GRU_LAYERS = 2

# Fusion
FUSION_DIM = 384  # 128 (CNN) + 256 (GRU)
FUSION_HEADS = 8

# Task heads
HEAD_HIDDEN = 128
HEAD_ATTENTION_HEADS = 4

# Regularization
DROPOUT_T0 = 0.2
DROPOUT_T1 = 0.3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
```

### Training Hyperparameters

```python
BATCH_SIZE = 48  # Smaller due to larger model
EPOCHS = 30
LEARNING_RATE = 5e-4  # Lower LR for stability
LR_SCHEDULER = "OneCycleLR"
PATIENCE = 10  # More patience for complex model

# Loss weights
LOSS_WEIGHT_T0 = 0.7
LOSS_WEIGHT_T1 = 0.3
MSE_REG_WEIGHT = 0.1  # For t1 only
```

### Data Augmentation (Optional)

Inspired by SOTA solutions:
1. **Random Masking**: Randomly mask 5% of features during training
2. **Gaussian Noise**: Add small noise (σ=0.01) to features
3. **Sequence Dropout**: Randomly drop 10% of timesteps (with caution for temporal consistency)

## Verification Plan

### Leakage-Free Validation

- ✅ Evaluate **ALL timesteps** from step 0 (no warm-up)
- ✅ Stateful inference matches training exactly
- ✅ Separate validation for t0 and t1 correlations

### Expected Performance
 
 | Metric | exp_06 | exp_07 Target | Improvement | Actual (Epoch 5) |
 |:---|:---|:---|:---|:---|
 | **Val Pearson** | 0.2577 | **0.28-0.30** | +0.02-0.04 | **0.3947** 🚀 |
 | t0 Correlation | 0.3707 | **0.39-0.41** | +0.02-0.04 | **0.3910** |
 | t1 Correlation | 0.1447 | **0.17-0.19** | +0.03-0.05 | **0.3984** 🤯 |
 
 **Rationale**:
 - Dual-head architecture should improve t1 significantly
 - Feature streams should capture complementary patterns
 - Pearson correlation loss should directly optimize metric

### Ablation Studies

To validate design choices:
1. **Baseline**: exp_06 architecture with exp_07 features
2. **+Dual-Stream**: Add dual-stream processing
3. **+Dual-Head**: Add dual-head architecture
4. **+Pearson Loss**: Replace MSE with Pearson loss
5. **Full exp_07**: All components

## Implementation Files

### File Structure

```
solutions/exp_07/
├── src/
│   ├── config.py           # Configuration
│   ├── features.py         # Leakage-free features (exp_06 + new)
│   ├── model.py            # DualHeadDualStream model
│   ├── losses.py           # Pearson correlation loss
│   ├── train.py            # Training script
│   └── utils.py            # Utilities
├── solution.py             # Stateful inference
└── best_model.pth          # Trained weights
```

### Key Implementation Details

**model.py**:
- [RawFeatureStream(nn.Module)](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_07/src/model.py#5-52): 1D-CNN for raw features
- [EngFeatureStream(nn.Module)](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_07/src/model.py#54-84): GRU for engineered features
- [FeatureFusion(nn.Module)](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_07/src/model.py#86-116): Attention-based fusion
- [TaskHead(nn.Module)](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_07/src/model.py#118-172): Task-specific head (t0 or t1)
- [DualHeadDualStream(nn.Module)](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_07/src/model.py#174-266): Main model

**losses.py**:
- [pearson_correlation_loss()](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_08/src/losses.py#5-51): Weighted Pearson loss
- [combined_loss()](file:///Users/ime/Work/WunderNN-Predictorium/solutions/exp_08/src/losses.py#53-92): t0 + t1 with different weights

**solution.py**:
- Stateful inference with separate streams
- Efficient NumPy feature computation
- Proper state management for GRU and CNN

## Risks and Mitigation

### Risk 1: Model Complexity → Overfitting

**Mitigation**:
- Higher dropout (0.2-0.3)
- Strong weight decay (1e-4)
- Early stopping (patience=10)
- Gradient clipping (1.0)
- Monitor train-val gap closely

### Risk 2: Longer Training Time

**Mitigation**:
- Start with 10-epoch test run
- Use smaller batch size (48 vs 64)
- Optimize data loading (num_workers=4)
- Consider mixed precision training

### Risk 3: Inference Speed

**Mitigation**:
- Profile inference time on validation set
- Target: <100ms per sequence (same as exp_06)
- Optimize CNN kernel sizes if needed
- Consider model pruning if too slow

## Success Criteria

### Minimum Viable Product (MVP)

- ✅ Validation score > 0.27 (better than exp_06's 0.2577)
- ✅ t1 correlation > 0.16 (significant improvement over 0.1447)
- ✅ No leakage (realistic scores, not inflated)
- ✅ Inference time < 100ms per sequence

### Stretch Goals

- 🎯 Validation score > 0.29 (approaching exp_02's 0.2691)
- 🎯 t0 correlation > 0.40
- 🎯 t1 correlation > 0.18
- 🎯 Leaderboard score > 0.30

## Timeline

1. **Implementation** (Day 1): Create all modules and test feature generation
2. **Short Training** (Day 1): 10-epoch test run to validate architecture
3. **Full Training** (Day 2): 30-epoch run with early stopping
4. **Evaluation** (Day 2): Analyze results, create submission
5. **Iteration** (Day 3+): Ablation studies if needed

## Next Steps

1. ✅ User review and approval of plan
2. Implement feature engineering (features.py)
3. Implement model architecture (model.py, losses.py)
4. Implement training script (train.py)
5. Implement solution.py for inference
6. Run 10-epoch test
7. Full 30-epoch training
8. Create submission and update experiment log
