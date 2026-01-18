# Experiment Tracking Log

| Exp ID | Description | Val Score | LB Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **exp_01** | PyTorch GRU (2 layers, 256 hidden), Raw Features, MSE Loss | 0.2310 | 0.2820 | Baseline PyTorch solution. Good generalization. |
| **exp_02** | PyTorch GRU + Feature Engineering (Mid-price, Spread, Imbalance, Diff) | 0.2691 | Pending | Added domain features (49 features). Improved Val score significantly. |
