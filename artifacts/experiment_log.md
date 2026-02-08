# Experiment Tracking Log

| Exp ID | Description | Val Score | LB Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **exp_01** | PyTorch GRU (2 layers, 256 hidden), Raw Features, MSE Loss | 0.2310 | 0.2820 | Baseline PyTorch solution. Good generalization. |
| **exp_02** | PyTorch GRU + Feature Engineering (Mid-price, Spread, Imbalance, Diff) | 0.2691 | Pending | Added domain features (49 features). Improved Val score significantly. |
| **exp_03** | CNN-GRU + Rolling Stats, Weighted MSE | 0.5153 | 0.1076 |  (t0: x, t1: x). |
| **exp_04** | CNN-GRU + Stationary Stats + SWA | 0.5153 | 0.1165 |  (t0: x, t1: x). |
| **exp_05** | Attention-GRU + CatBoost | 0.5153 | 0.1076 |  (t0: x, t1: x). |
| **exp_06** | Unidirectional GRU + Leakage-Free Features (65 feats), Weighted MSE | 0.2577 | 0.2634 | Fixed leakage from exp_03-05. Clean validation (t0: 0.3707, t1: 0.1447). |

