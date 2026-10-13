# exp_12: Specialized t1 Prediction Model

## Motivation

From experiment results:
- t0 prediction: ~0.37 Pearson (decent)
- t1 prediction: ~0.14 Pearson (much harder)

t1 is a longer-horizon prediction, likely requiring:
- Different features (more context)
- Different model capacity
- Different regularization

## Strategy

### Approach 1: Separate Models
Train two completely separate models:
- Model A: Optimized for t0 only
- Model B: Optimized for t1 only (larger context, different architecture)

### Approach 2: Multi-Task with Task-Specific Layers
- Shared backbone (GRU)
- Separate, deeper heads for each target
- Different loss weights (focus more on t1)

### Approach 3: Cascaded Prediction
- First predict t0
- Use t0 prediction as additional feature for t1
- May capture dependency between targets

## Implementation Plan

We'll try **Approach 2** first (most practical):

```
Input (batch, seq, 32 features)
        ↓
[GRU Backbone] → shared representation
        ↓
    ┌───┴───┐
    ↓       ↓
[t0 Head]  [t1 Head]
(shallow)  (deeper, more capacity)
    ↓       ↓
  t0_pred  t1_pred
```

## t1-Specific Modifications

1. **Deeper t1 head**: 2 hidden layers instead of 1
2. **More features for t1**: Add lagged predictions, rolling stats
3. **Higher weight on t1 loss**: 0.4 t0 + 0.6 t1 (reverse of usual)
4. **Larger context window**: May need to see more history

## Hyperparameters

- GRU hidden: 256 (same as exp_01)
- t0 head: 1 hidden layer (64 units)
- t1 head: 2 hidden layers (128 → 64 units)
- t1 dropout: 0.4 (higher regularization)
- Loss weights: t0=0.4, t1=0.6
