# Experiment Tracking Log

| Exp ID | Description | Val Score | LB Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **exp_01** | PyTorch GRU (2 layers, 256 hidden), Raw Features, MSE Loss | 0.2310 | 0.2820 | Baseline PyTorch solution. Good generalization. |
| **exp_02** | PyTorch GRU + Feature Engineering (Mid-price, Spread, Imbalance, Diff) | 0.2691 | 0.2766 | Added domain features (49 features). Improved Val score significantly. |
| **exp_03** | CNN-GRU + Rolling Stats, Weighted MSE | 0.5153 | 0.1076 |  (t0: x, t1: x). |
| **exp_04** | CNN-GRU + Stationary Stats + SWA | 0.5153 | 0.1165 |  (t0: x, t1: x). |
| **exp_05** | Attention-GRU + CatBoost | 0.5153 | 0.1076 |  (t0: x, t1: x). |
| **exp_06** | Unidirectional GRU + Leakage-Free Features (65 feats), Weighted MSE | 0.2577 | 0.2634 | Fixed leakage from exp_03-05. Clean validation (t0: 0.3707, t1: 0.1447). |
| **exp_07** | Dual-Head Dual-Stream (CNN+GRU), Pearson Loss | 0.3947 | TIMEOUT | Inflated val score suggests leakage in additional features. Timeout (70+ min) due to pandas per-step. |
| **exp_09** | Dual-Head GRU + Pearson Loss (exp_06 features) | Pending | Pending | Combines exp_07's dual-head + Pearson loss with exp_06's leakage-free features and fast NumPy inference. |
| **exp_10** | DeepLOB-style CNN-GRU (Inception + GRU, 32 raw features) | 0.5952 | Pending | t0: 0.7802, t1: 0.4101. Very high val score - likely has symmetric padding leakage in CNN. Best at epoch 13. |
| **exp_11** | - | Pending | Pending | - |
| **exp_12** | - | Pending | Pending | - |
| **exp_13** | - | Pending | Pending | - |
| **exp_14** | Catboost (66 features) | 0.1975 | Pending |  t0: 0.3096 ± 0.0001, t1: 0.0853 ± 0.0004 |
| **exp_15** | Catboost (100 features) | 0.1980 | Pending | t0: 0.3130 ± 0.0002, t1: 0.0830 ± 0.0007 |
| **exp_16** | Catboost (X features) | Pending | Pending | - |
| **exp_17** | - | Pending | Pending | - |
| **exp_18** | - | Pending | Pending | - |
| **exp_19** | - | Pending | Pending | - |