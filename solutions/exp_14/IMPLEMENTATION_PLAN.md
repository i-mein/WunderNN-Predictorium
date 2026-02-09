# Experiment 14: CatBoost Baseline

## Objective
Establish a strong GBDT baseline using CatBoost with the same 65 leakage-free features from exp_06.

## Hypothesis
- GBDT models may capture non-linear feature interactions more efficiently
- Separate models for t0 and t1 allow independent optimization
- Faster training than DL approaches enables rapid iteration

## Architecture

### Model
- **Type**: CatBoostRegressor (separate models for t0 and t1)
- **Loss**: RMSE with Pearson-optimized hyperparameters
- **Features**: 65 features (32 raw + 16 base + 12 rolling + 5 diffs)

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| iterations | 3000 | Sufficient for convergence |
| depth | 8 | Balance complexity vs generalization |
| learning_rate | 0.03 | Standard for GBDT |
| l2_leaf_reg | 3.0 | Regularization |
| subsample | 0.8 | Prevent overfitting |
| random_strength | 0.5 | Add randomness |
| bagging_temperature | 0.0 | Standard bagging |

### Training Strategy
- Train separate models for t0 and t1
- Weight early steps (0-99) vs late steps (100-999) with 2:1 ratio
- Monitor Pearson correlation on validation set
- Early stopping with patience 200

## Feature Set (65 features)

### Raw Features (24)
- p0-p11: Price levels
- v0-v11: Volume levels

### Base Engineered (16)
- mid_price_0, mid_price_1
- spread_0, spread_1
- vol_imbal_0 through vol_imbal_5
- total_bid_vol, total_ask_vol, vol_imbal_total
- relative_spread_0, vol_ratio, price_depth

### Rolling Features (12)
- mid_price_0_mean_3, mid_price_0_std_3, mid_price_0_mean_10, mid_price_0_std_10
- spread_0_mean_3, spread_0_std_3, spread_0_mean_10, spread_0_std_10
- vol_imbal_total_mean_3, vol_imbal_total_std_3, vol_imbal_total_mean_10, vol_imbal_total_std_10

### Diff Features (5)
- p0_diff, p6_diff, mid_price_0_diff, spread_0_diff, vol_imbal_total_diff

## Expected Metrics
- Val Pearson: 0.15-0.20 (baseline expectation)
- Training time: ~10-20 minutes

## File Structure
```
exp_14/
├── IMPLEMENTATION_PLAN.md
├── solution.py          # Inference code
├── model_catboost_t0.cbm # Trained model for t0
├── model_catboost_t1.cbm # Trained model for t1
└── src/
    ├── config.py        # Hyperparameters
    ├── features.py      # Feature generation
    └── train.py         # Training script
```

## Run Commands
```bash
# Training
python solutions/exp_14/src/train.py

# Evaluation (uses exp_01 run_eval.sh)
./scripts/run_eval.sh exp_14
```

## Next Steps After Baseline
- exp_15: Add micro-price, LOB pressure, trade flow features
- exp_16: Hyperparameter tuning with Optuna
- exp_17: Multi-target ensemble
