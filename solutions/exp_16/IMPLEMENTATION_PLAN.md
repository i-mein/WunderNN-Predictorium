# Experiment 16: CatBoost with Temporal/Momentum Features + Optuna

## Objective
Improve t1 prediction with temporal/momentum features and separate hyperparameter optimization.

## Key Changes from exp_15

### New Temporal/Momentum Features (~25 new features)

#### Longer Rolling Windows
| Feature | Description |
|---------|-------------|
| `*_roll_mean_20`, `*_roll_std_20` | 20-step rolling for mid_price, vol_imbal, micro_price |
| `*_roll_mean_50`, `*_roll_std_50` | 50-step rolling for key features |
| `*_roll_range_20`, `*_roll_range_50` | Rolling range (max-min) as volatility proxy |

#### Momentum Indicators
| Feature | Description |
|---------|-------------|
| `momentum_5`, `momentum_10`, `momentum_20` | Price momentum (current - lagged) |
| `vol_momentum_5`, `vol_momentum_10` | Volume imbalance momentum |
| `micro_momentum_5`, `micro_momentum_10` | Micro-price deviation momentum |

#### Acceleration Features (2nd Order Diffs)
| Feature | Description |
|---------|-------------|
| `mid_price_0_diff_accel` | Acceleration of mid-price change |
| `vol_imbal_total_diff_accel` | Acceleration of volume imbalance |
| `micro_price_deviation_diff_accel` | Acceleration of micro-price deviation |

#### Trend Strength Indicators
| Feature | Description |
|---------|-------------|
| `trend_strength_10`, `trend_strength_20` | Momentum / volatility (Sharpe-like) |
| `vol_trend_strength` | Volume trend strength |
| `price_position_20`, `price_position_50` | Position within recent range |

### Separate Hyperparameter Optimization

**t0 Search Space** (allows more complexity):
- iterations: 2000-5000
- learning_rate: 0.02-0.1
- depth: 6-10
- l2_leaf_reg: 1.0-10.0

**t1 Search Space** (more regularization):
- iterations: 500-2000
- learning_rate: 0.005-0.05
- depth: 4-8
- l2_leaf_reg: 3.0-20.0
- subsample: 0.5-0.9
- min_data_in_leaf: 10-100

## Expected Improvement
- t1 Pearson: 0.085 → 0.10+ (target)
- Combined: 0.198 → 0.21+ (target)

## Run Commands
```bash
# Run Optuna optimization (single seed)
python solutions/exp_16/src/optimize.py
```

## Total Feature Count
- exp_15: ~100 features
- exp_16: ~125 features (+25 temporal/momentum)
