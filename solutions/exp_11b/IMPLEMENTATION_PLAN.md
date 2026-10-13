# Exp 11b: Dual Transformer Models for t0/t1

## Concept
Instead of a single transformer with dual heads, use **two separate transformer models**:
- **Model T0**: Optimized for t0 prediction (easier, ~0.27 Pearson)
- **Model T1**: Optimized for t1 prediction (harder, ~0.08 Pearson)

## Key Differences from exp_11
1. Two completely separate models (not shared backbone)
2. Different hyperparameters tuned for each task
3. More epochs (50+) since exp_11 showed slow but steady improvement

## Architecture

### Model T0 (Simpler - t0 is easier)
- d_model: 48
- n_heads: 4
- n_layers: 2
- d_ff: 96
- dropout: 0.2

### Model T1 (Deeper - t1 is harder)
- d_model: 96
- n_heads: 6  
- n_layers: 4
- d_ff: 256
- dropout: 0.4

## Training Strategy
- Train both models simultaneously
- Early stopping based on average Pearson
- Higher learning rate for t0 (converges faster)
- Lower learning rate for t1 (needs more careful optimization)

## Expected Benefits
1. Each model can specialize for its target
2. No competition for model capacity between tasks
3. Different regularization for different difficulty levels
