# Experiment 15: Enhanced CatBoost with Micro-Price & Trade Flow

## Objective
Improve upon exp_14 baseline by adding domain-specific features for order book dynamics.

## Hypothesis
- Micro-price features capture volume-weighted fair value better than simple mid-price
- Trade flow / LOB pressure features capture directional momentum
- These features should improve t1 prediction (longer horizon)

## New Features Added

### Micro-Price Features (5)
| Feature | Description |
|---------|-------------|
| `micro_price` | Volume-weighted mid: (p0*v6 + p6*v0) / (v0+v6) |
| `micro_price_deviation` | micro_price - mid_price_0 |
| `micro_price_deviation_norm` | Deviation normalized by spread |
| `weighted_mid_2` | Level 2 micro-price |
| `micro_price_deviation_2` | Level 2 deviation |

### Order Book Pressure Features (12)
| Feature | Description |
|---------|-------------|
| `bid_pressure_0` | v0 / total_bid_vol (concentration at best) |
| `bid_pressure_01` | (v0+v1) / total_bid_vol |
| `ask_pressure_0` | v6 / total_ask_vol |
| `ask_pressure_01` | (v6+v7) / total_ask_vol |
| `bid_slope` | Price impact per unit bid volume |
| `ask_slope` | Price impact per unit ask volume |
| `slope_imbal` | (bid_slope - ask_slope) / (bid_slope + ask_slope) |
| `depth_imbal_near` | Near-level volume imbalance |
| `depth_imbal_far` | Far-level volume imbalance |
| `bid_vwap` | Volume-weighted average bid price |
| `ask_vwap` | Volume-weighted average ask price |
| `vwap_spread` | ask_vwap - bid_vwap |

### Price Level Features (6)
| Feature | Description |
|---------|-------------|
| `price_range` | Full depth price range (p11 - p0) |
| `bid_range` | Bid side range (p5 - p0) |
| `ask_range` | Ask side range (p11 - p6) |
| `bid_spacing` | Average bid tick spacing |
| `ask_spacing` | Average ask tick spacing |
| `spacing_ratio` | bid_spacing / ask_spacing |

### Additional Rolling Features (8)
- Rolling mean/std for `micro_price_deviation` (windows 3, 10)
- Rolling mean/std for `depth_imbal_near` (windows 3, 10)

### Additional Diff Features (3)
- `micro_price_diff`
- `micro_price_deviation_diff`
- `depth_imbal_near_diff`

## Total Feature Count
- **exp_14 baseline**: 65 features
- **exp_15 enhanced**: ~96 features (+31 new)

## Expected Metrics
- Val Pearson: 0.20-0.22 (targeting improvement over exp_14's 0.1975)
- Particularly hoping to improve t1

## Run Commands
```bash
# Multi-seed training
python solutions/exp_15/src/train.py --multi-seed
```
