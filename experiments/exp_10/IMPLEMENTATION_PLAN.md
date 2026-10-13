# Implementation Plan: Experiment 10 - Improved GRU with Regularization

## Goal

Improve upon exp_06 (Val: 0.2577, LB: 0.2634) and exp_09 (Val: 0.2057) by addressing:
1. **Overfitting in exp_09** - t1 correlation collapsed from 0.09 to 0.02 during training
2. **t1 instability** - t1 is much harder to predict, needs special treatment
3. **Model capacity vs regularization** - Find the right balance

## Analysis of Previous Experiments

### exp_01 (Best LB: 0.2820)
- Simple GRU with 32 raw features
- No feature engineering → generalizes well to test
- Gap: Val 0.2310 < LB 0.2820 (unusual, suggests val is harder)

### exp_06 (LB: 0.2634)
- 65 engineered features (rolling stats, diffs)
- Weighted MSE loss (0.7 * t0 + 0.3 * t1)
- Good t0 (0.3707), weak t1 (0.1447)

### exp_09 (Val: 0.2057)
- Dual-head with Pearson loss
- **Problem**: Severe overfitting, t1 collapsed
- Training curves show divergence after epoch 8

### Key Observations
1. **Raw features generalize better** (exp_01 > exp_06 on LB)
2. **t1 is extremely difficult** - all models struggle
3. **Overfitting is the main enemy** - exp_09 overfit badly
4. **Simpler is better** - exp_01's simplicity won on LB

## Hypothesis for exp_10

**Core Insight**: The gap between exp_01 (LB 0.2820) and exp_06 (LB 0.2634) suggests that engineered features may hurt generalization. But exp_01's val score (0.2310) was lower than exp_06's (0.2577).

**Strategy**: Combine the best of both:
1. Use **minimal, high-quality features** (not 65, maybe 40-50)
2. Apply **strong regularization** to prevent overfitting
3. Use **separate treatment for t0 vs t1** (different regularization)
4. Add **auxiliary techniques** to stabilize t1

## Architecture: Regularized Dual-Head GRU

### Key Changes from exp_09

| Component | exp_09 | exp_10 | Rationale |
|-----------|--------|--------|-----------|
| Features | 65 | **45** (pruned) | Remove potentially harmful features |
| GRU Layers | 2 | **2** (same) | Keep backbone stable |
| Hidden Dim | 256 | **192** | Slightly smaller to reduce overfitting |
| Head Hidden | 128 | **64** | Smaller heads |
| Dropout (backbone) | 0.3 | **0.4** | More regularization |
| Dropout (t0 head) | 0.2 | **0.3** | More regularization |
| Dropout (t1 head) | 0.3 | **0.5** | Much more for unstable t1 |
| Weight Decay | 1e-4 | **1e-3** | Stronger L2 regularization |
| Label Smoothing | No | **Yes** | Reduce overconfidence |

### Feature Selection (45 features)

**Keep** (high importance, low leakage risk):
- Raw prices/volumes: p0-p11, v0-v11 (24)
- Mid prices: mid_price_0, mid_price_1 (2)
- Spreads: spread_0, spread_1 (2)
- Volume imbalances: vol_imbal_0-5, vol_imbal_total (7)
- Basic ratios: vol_ratio, relative_spread_0 (2)
- Price diffs: dp0-dp3 (4)
- Volume diffs: dv0-dv3 (4)

**Remove** (potential leakage or noise):
- Rolling statistics (mean, std) - may leak future info
- Complex engineered features (micro_price, depth_imbal, etc.)

### Model Architecture

```
Input (45 features)
        ↓
   LayerNorm
        ↓
  GRU (192 hidden, 2 layers, dropout=0.4)
        ↓
  ┌─────┴─────┐
  │           │
LayerNorm   LayerNorm
  │           │
t0 Head    t1 Head
  │           │
FC(192→64)  FC(192→64)
GELU        GELU
Dropout(0.3) Dropout(0.5)
FC(64→1)    FC(64→1)
  │           │
  ↓           ↓
 t0          t1
```

### Loss Function

**Smooth L1 Loss (Huber)** instead of MSE:
- More robust to outliers
- Prevents gradient explosion on hard samples

```python
def smooth_l1_weighted(pred_t0, pred_t1, target_t0, target_t1):
    loss_t0 = F.smooth_l1_loss(pred_t0, target_t0, beta=0.5)
    loss_t1 = F.smooth_l1_loss(pred_t1, target_t1, beta=0.5)
    
    # Weight t0 more (easier, more reliable)
    return 0.6 * loss_t0 + 0.4 * loss_t1
```

### Regularization Techniques

1. **Gradient Clipping**: max_norm=0.5 (tighter than exp_09's 1.0)
2. **Weight Decay**: 1e-3 (10x stronger than exp_09)
3. **Dropout**: Higher throughout (0.4-0.5)
4. **Early Stopping**: patience=5 (shorter, stop sooner)
5. **Learning Rate**: Start lower (5e-4 instead of 1e-3)
6. **Label Smoothing**: Add small noise to targets

### Data Augmentation

**Gaussian Noise Injection**:
```python
def add_noise(x, std=0.01):
    if self.training:
        noise = torch.randn_like(x) * std
        return x + noise
    return x
```

## Training Configuration

```python
# Model
INPUT_DIM = 45
HIDDEN_DIM = 192
NUM_LAYERS = 2
HEAD_HIDDEN = 64
DROPOUT_BACKBONE = 0.4
DROPOUT_T0 = 0.3
DROPOUT_T1 = 0.5

# Training
BATCH_SIZE = 64
EPOCHS = 20  # Fewer epochs, less overfitting
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
GRADIENT_CLIP = 0.5

# Loss
LOSS_WEIGHT_T0 = 0.6
LOSS_WEIGHT_T1 = 0.4
HUBER_BETA = 0.5

# Early stopping
PATIENCE = 5

# Augmentation
NOISE_STD = 0.01
```

## Alternative Experiment: exp_10b (Raw Features Only)

Given exp_01's strong LB performance with raw features, also try:

```python
# exp_10b: Return to raw features like exp_01
INPUT_DIM = 32  # Only p0-p11, v0-v11, dp0-dp3, dv0-dv3
HIDDEN_DIM = 256
# But with dual-head architecture and better regularization
```

## Expected Performance

| Metric | exp_06 | exp_09 | exp_10 Target |
|--------|--------|--------|---------------|
| Val Pearson | 0.2577 | 0.2057 | **0.24-0.26** |
| t0 Correlation | 0.3707 | 0.3150 | **0.35-0.37** |
| t1 Correlation | 0.1447 | 0.0963 | **0.12-0.15** |
| LB Score | 0.2634 | Pending | **0.27-0.28** |

## File Structure

```
solutions/exp_10/
├── IMPLEMENTATION_PLAN.md
├── src/
│   ├── config.py        # Configuration
│   ├── features.py      # Simplified features (45)
│   ├── model.py         # Regularized Dual-Head GRU
│   ├── train.py         # Training with augmentation
│   └── utils.py         # Utilities
├── solution.py          # Inference
└── best_model.pth       # Trained weights
```

## Success Criteria

### Must Have
- ✅ No overfitting (val score stable, not dropping)
- ✅ t1 correlation stays positive throughout training
- ✅ Val score > 0.22

### Nice to Have
- 🎯 Val score > exp_06 (0.2577)
- 🎯 LB score > exp_01 (0.2820)
- 🎯 t1 stable above 0.10

## Experiments to Run

1. **exp_10**: Regularized dual-head with 45 features
2. **exp_10b**: Regularized dual-head with 32 raw features (like exp_01)

If exp_10b performs better, it confirms that feature engineering hurts generalization for this task.
