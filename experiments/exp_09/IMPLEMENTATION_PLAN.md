# Implementation Plan: Experiment 09 - Dual-Head GRU with Pearson Loss

## Goal

Build exp_09 that resolves the issues identified in exp_07:
1. **Remove data leakage** - Use exp_06's proven leakage-free features (65 features)
2. **Fix timeout** - Use exp_06's fast NumPy inference (no pandas per step)
3. **Keep improvements** - Maintain dual-head architecture and Pearson loss from exp_07

## Background Analysis

### Exp_07 Issues

| Issue | Root Cause | Impact |
|:---|:---|:---|
| **Val Score > 0.39** | Additional rolling features (min/max/range) may leak information or cause different patterns | Inflated validation, poor generalization |
| **Timeout (70+ min)** | Pandas DataFrame operations per timestep in solution.py | Inference too slow for submission |

### Exp_06 Success Factors

| Factor | Details | Benefit |
|:---|:---|:---|
| **65 Features** | Proven leakage-free feature set | Realistic validation scores |
| **NumPy Inference** | Pure NumPy feature computation | ~9 min submission runtime |
| **Simple GRU** | Single backbone, efficient | Fast and stable |

### Key Changes in Exp_09

| Component | exp_06 | exp_07 | exp_09 |
|:---|:---|:---|:---|
| Features | 65 (leakage-free) | 75 (potential leakage) | **65 (from exp_06)** |
| Model | Single-head GRU | Dual-head CNN+GRU | **Dual-head GRU** |
| Loss | Weighted MSE | Pearson + MSE | **Pearson + MSE** |
| Inference | NumPy | Pandas per step | **NumPy (from exp_06)** |

## Architecture: Dual-Head GRU

### Design

```
Input (65 features)
        ↓
  Shared GRU Backbone
    (256 hidden, 2 layers)
        ↓
  ┌─────┴─────┐
  │           │
t0 Head    t1 Head
  │           │
FC(256→128)  FC(256→128)
ReLU         ReLU
Dropout(0.2) Dropout(0.3)
FC(128→1)    FC(128→1)
  │           │
  ↓           ↓
 t0          t1
```

### Key Design Decisions

1. **Shared Backbone**: Same GRU as exp_06 - proven efficient
2. **Separate Heads**: Allow task-specific optimization (t0 vs t1)
3. **Different Dropout**: t0 (0.2) - easier to predict; t1 (0.3) - harder, needs more regularization
4. **No CNN/Attention**: Keep inference simple and fast

## Loss Function

### Pearson Correlation Loss

```python
def pearson_correlation_loss(y_pred, y_true, eps=1e-8):
    """Negative Pearson correlation as loss."""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    pred_mean = y_pred.mean()
    true_mean = y_true.mean()
    
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    
    cov = (pred_centered * true_centered).mean()
    pred_std = torch.sqrt((pred_centered ** 2).mean() + eps)
    true_std = torch.sqrt((true_centered ** 2).mean() + eps)
    
    corr = cov / (pred_std * true_std)
    return -corr
```

### Combined Loss

```python
# Per-target losses
loss_t0 = pearson_loss(pred_t0, target_t0) + 0.1 * mse_loss(pred_t0, target_t0)
loss_t1 = pearson_loss(pred_t1, target_t1) + 0.1 * mse_loss(pred_t1, target_t1)

# Combined (equal weights, since we want to optimize both targets)
total_loss = 0.5 * loss_t0 + 0.5 * loss_t1
```

## Training Configuration

```python
# Model
INPUT_DIM = 65  # Same as exp_06
HIDDEN_DIM = 256
NUM_LAYERS = 2
HEAD_HIDDEN = 128
DROPOUT_T0 = 0.2
DROPOUT_T1 = 0.3

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Loss
LOSS_WEIGHT_T0 = 0.5
LOSS_WEIGHT_T1 = 0.5
MSE_REG_WEIGHT = 0.1

# Early stopping
PATIENCE = 10
```

## Expected Performance

| Metric | exp_06 | exp_07 | exp_09 Target |
|:---|:---|:---|:---|
| Val Pearson | 0.2577 | 0.3947 (leakage) | **0.26-0.28** |
| t0 Correlation | 0.3707 | 0.3910 | **0.37-0.39** |
| t1 Correlation | 0.1447 | 0.3984 (leakage) | **0.15-0.17** |
| LB Score | 0.2634 | TIMEOUT | **0.27-0.28** |
| Runtime | 9 min | 70+ min | **~9 min** |

## Verification Plan

### Leakage Check
- Val score should be in range 0.25-0.30 (similar to exp_06)
- t1 should be harder than t0 (t1 < t0)
- LB score should be close to val score

### Inference Speed
- Use NumPy-only feature computation (from exp_06)
- No pandas DataFrame operations per step
- Target runtime: <15 min

## File Structure

```
solutions/exp_09/
├── src/
│   ├── config.py     # Configuration
│   ├── features.py   # IDENTICAL to exp_06
│   ├── model.py      # DualHeadGRU
│   ├── losses.py     # Pearson correlation loss
│   ├── train.py      # Training script
│   └── utils.py      # Utilities
├── solution.py       # NumPy inference (based on exp_06)
└── best_model.pth    # Trained weights
```

## Success Criteria

### Must Have
- ✅ Val score < 0.35 (no leakage)
- ✅ Submission completes within 60 min
- ✅ LB score close to val score

### Nice to Have
- 🎯 Val score > exp_06 (0.2577)
- 🎯 LB score > exp_06 (0.2634)
- 🎯 t1 improvement over exp_06
