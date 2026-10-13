# Experiment Plan: Dual-Path Modeling Strategy

## Overview

Split into two parallel modeling paths, then ensemble for best results:

1. **Path A: Deep Learning** - Advanced architectures using raw features only
2. **Path B: GBDT** - Feature engineering with CatBoost/LightGBM

---

## Path A: Deep Learning (Raw Features Only)

### Baseline
- **exp_06**: GRU + shared head, 65 features, MSE loss → Val 0.2577

### Experiments

#### exp_10: Transformer Encoder
**Hypothesis**: Self-attention can capture long-range dependencies better than GRU
```
Architecture:
- Raw 32 features only (no engineered features)
- Positional encoding (learnable or sinusoidal)
- Transformer encoder (4 layers, 4 heads, d_model=128)
- Dual-head output (separate t0/t1 heads)
- Layer normalization + residual connections

Training:
- MSE loss (proven stable)
- LR: 1e-4 with warmup
- Batch size: 32 (smaller for attention memory)
- Gradient clipping: 1.0
```

#### exp_11: GRU + Temporal Attention
**Hypothesis**: Attention over GRU states can focus on relevant timesteps
```
Architecture:
- Raw 32 features
- Bidirectional GRU (2 layers, hidden=128)
- Temporal attention layer (query from last state)
- Dual-head output

Training:
- MSE loss
- LR: 5e-4
```

#### exp_12: Hybrid CNN + GRU
**Hypothesis**: CNN extracts local patterns, GRU captures sequence dynamics
```
Architecture:
- Raw 32 features
- 1D Conv layers (kernel sizes: 3, 5, 7)
- Max pooling
- GRU on top of CNN features
- Dual-head output
```

#### exp_13: WaveNet-style Dilated Convolutions
**Hypothesis**: Dilated convolutions efficiently capture multi-scale temporal patterns
```
Architecture:
- Raw 32 features
- Dilated causal convolutions (dilations: 1, 2, 4, 8, 16)
- Skip connections
- Dual-head output
```

### Feature Ablation Study
Test performance with different raw feature subsets:
- Bid prices only (4 features)
- Ask prices only (4 features)
- Bid/Ask prices (8 features)
- Bid/Ask + volumes (16 features)
- Full raw (32 features)

---

## Path B: GBDT + Feature Engineering

### Baseline
- **exp_05**: CatBoost + DL ensemble, hand-crafted features

### Feature Engineering Strategy

#### Level 1: Basic Features (from exp_06)
```python
# Already implemented - 65 features
- Mid prices (bid+ask)/2 for each level
- Spreads (ask-bid) for each level
- Volume imbalances (bid_vol - ask_vol)
- Rolling means/stds (windows: 3, 10)
- Diffs from previous timestep
```

#### Level 2: Advanced LOB Features
```python
# Order book imbalance metrics
- Weighted mid price (volume-weighted)
- Order imbalance ratio: sum(bid_vol) / sum(ask_vol)
- Depth imbalance: cumsum(bid_vol) vs cumsum(ask_vol) at each level
- Price pressure: (bid_vol[0] - ask_vol[0]) / (bid_vol[0] + ask_vol[0])

# Cross-level features
- Price slope: regression coefficient of price vs level
- Volume concentration: vol[0] / sum(vol)
- Level correlations

# Microstructure features
- Realized volatility (rolling)
- Quote intensity changes
- Spread relative to mid price
```

#### Level 3: Lag Features
```python
# Careful: must use groupby('seq_ix') to avoid leakage
- Lag features: feature_t-1, feature_t-2, feature_t-3
- Lag diffs: feature_t - feature_t-1
- Rolling stats on lags
```

### Experiments

#### exp_14: CatBoost with Level 1 Features
**Baseline GBDT with proven features**
```python
Model:
- CatBoostRegressor (separate models for t0, t1)
- iterations: 1000
- learning_rate: 0.05
- depth: 6
- l2_leaf_reg: 3
- random_seed: 42

Features:
- Same 65 features as exp_06
- No sequence modeling (each timestep independent)
```

#### exp_15: CatBoost with Level 2 Features
**Add advanced LOB features**
```python
New features (~30 additional):
- Weighted mid prices
- Order/depth imbalances
- Price pressure indicators
- Volume concentration
- Cross-level ratios

Total: ~95 features
```

#### exp_16: CatBoost with Lag Features
**Add temporal context via lag features**
```python
New features:
- Lag-1, Lag-2, Lag-3 of key features (mid_price, spread, imbalance)
- Lag diffs
- EMA features

Total: ~150 features
```

#### exp_17: LightGBM Comparison
**Compare GBDT implementations**
```python
Model:
- LGBMRegressor
- Similar hyperparameters to CatBoost
- Native categorical handling if needed
```

#### exp_18: CatBoost with Target Encoding
**Careful: potential leakage risk**
```python
- Target-encode categorical bins of features
- Use proper CV to avoid leakage
- seq_ix grouped cross-validation
```

---

## Path C: Ensemble Strategies

### exp_19: Simple Average Ensemble
```python
pred = 0.5 * dl_pred + 0.5 * gbdt_pred
```

### exp_20: Weighted Ensemble (Optimized)
```python
# Optimize weights on validation set
from scipy.optimize import minimize

def objective(weights):
    pred = weights[0] * dl_pred + weights[1] * gbdt_pred
    return -pearson_correlation(pred, target)

result = minimize(objective, [0.5, 0.5], bounds=[(0,1), (0,1)])
```

### exp_21: Stacking Ensemble
```python
# Meta-learner on base model predictions
features = [dl_pred_t0, dl_pred_t1, gbdt_pred_t0, gbdt_pred_t1]
meta_model = Ridge()  # Simple meta-learner to avoid overfitting
```

### exp_22: Target-Specific Ensemble
```python
# Different weights for t0 vs t1
# t0 is easier (higher correlation) - maybe GBDT better
# t1 is harder (lower correlation) - maybe DL better
pred_t0 = w0_dl * dl_t0 + w0_gbdt * gbdt_t0
pred_t1 = w1_dl * dl_t1 + w1_gbdt * gbdt_t1
```

---

## Experiment Priority Order

### Phase 1: Parallel Development
1. **exp_14** (CatBoost baseline) - Quick, establishes GBDT baseline
2. **exp_10** (Transformer) - Test attention mechanism
3. **exp_15** (CatBoost + advanced features) - Feature engineering

### Phase 2: Refinement
4. **exp_16** (CatBoost + lags) - Temporal features for GBDT
5. **exp_11** (GRU + attention) - Simpler attention approach
6. **exp_17** (LightGBM) - Alternative GBDT

### Phase 3: Ensemble
7. **exp_19** (Simple ensemble) - Quick win
8. **exp_20** (Weighted ensemble) - Optimize weights
9. **exp_22** (Target-specific) - Fine-tune per target

---

## Key Considerations

### Avoiding Leakage
- Always use `groupby('seq_ix')` for rolling/lag features
- Never use future information within a sequence
- Validation should use separate sequences (already done)

### Inference Speed
- Target: < 10 minutes for submission
- GBDT: Very fast inference
- DL: Use NumPy-only inference (no pandas per step)
- Avoid: Complex feature computation per timestep

### Model Complexity vs Performance
| Model | Complexity | Expected Val | Inference Speed |
|-------|------------|--------------|-----------------|
| GRU (exp_06) | Medium | 0.26 | Fast |
| Transformer | High | 0.25-0.28? | Medium |
| CatBoost | Low | 0.24-0.27? | Very Fast |
| Ensemble | Medium | 0.27-0.30? | Medium |

---

## Current Status

| Experiment | Status | Val Pearson | Notes |
|------------|--------|-------------|-------|
| exp_06 | ✅ Done | 0.2577 | GRU baseline |
| exp_07 | ❌ Failed | 0.3947 (leak) | Timeout + leakage |
| exp_09 | 🔄 Running | TBD | Dual-head MSE |
| exp_10+ | 📋 Planned | - | See above |

---

## Next Steps After exp_09

1. Analyze exp_09 results
2. If exp_09 > exp_06: Use dual-head as DL baseline
3. Start exp_14 (CatBoost) in parallel with exp_10 (Transformer)
4. Iterate based on results
