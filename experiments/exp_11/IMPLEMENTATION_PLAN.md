# exp_11: Causal Transformer for LOB Prediction

## Architecture Overview

Based on "Transformers for Limit Order Books" (arXiv:2003.00130):
- Causal self-attention (masked to prevent future leakage)
- Positional encoding for sequence awareness
- Dual heads for t0/t1 prediction

## Key Design Choices

1. **Causal Masking**: Upper triangular mask in attention to prevent seeing future
2. **Raw 32 features**: Following exp_01's success with raw features
3. **Lightweight**: Small model to prevent overfitting
4. **No CNN**: Pure attention-based (CNN didn't help in exp_10b)

## Architecture

```
Input (batch, seq, 32 features)
        ↓
[Linear Projection] → (batch, seq, d_model=64)
        ↓
[Positional Encoding]
        ↓
[Causal Transformer Block x 2]
  - Multi-Head Self-Attention (4 heads, causal mask)
  - Feed-Forward Network
  - LayerNorm + Residual
        ↓
[Dual Heads]
  - t0_head: Linear → output
  - t1_head: Linear → output (more dropout)
```

## Hyperparameters

- d_model: 64
- n_heads: 4
- n_layers: 2
- d_ff: 128
- dropout: 0.3
- learning_rate: 3e-4
- weight_decay: 1e-3

## Training Strategy

- Huber loss (robust to outliers)
- Early stopping on validation Pearson
- Gradient clipping (0.5)
- Cosine annealing LR schedule
