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
| **exp_10** | DeepLOB-style CNN-GRU (Inception + GRU, 32 raw features) | 0.5952 | 0.1005 | t0: 0.7802, t1: 0.4101. Symmetric padding leakage in CNN confirmed by poor LB. |
| **exp_10b** | DeepLOB-style CNN-GRU with CAUSAL convolutions | 0.1664 | Pending | t0: 0.2671, t1: 0.0657. Causal fix worked - realistic val score now. |
| **exp_11** | - | Pending | Pending | - |
| **exp_12** | - | Pending | Pending | - |
| **exp_13** | Hybrid Transformer-GRU (d=128, 2 layers, 7 eng features) | 0.2115 | **0.2262** | t0: 0.3221, t1: 0.1010. Single-step inference. Good LB generalization! |
| **exp_14** | Catboost (66 features) | 0.1975 | Pending |  t0: 0.3096 ± 0.0001, t1: 0.0853 ± 0.0004 |
| **exp_15** | Catboost (100 features) | 0.1980 | Pending | t0: 0.3130 ± 0.0002, t1: 0.0830 ± 0.0007 |
| **exp_16** | Catboost (X features) | Pending | Pending | - |
| **exp_17** | Enhanced Hybrid (d=160, 3 layers, DeltaProcessor, 10 eng features) | 0.3362 | 0.1679 | t0: 0.3129, t1: 0.3595. Overfit on t1 - high val but poor LB. Model too complex. |
| **exp_18** | Multi-Stream GRU (4 streams: price/vol/dprice/dvol, 154K params) | 0.2196 | Pending | t0: 0.3292, t1: 0.1100. Separate GRU per feature category, leakage-free eng features. |
| **exp_19** | Conv2D LOB (DeepLOB-inspired, 123 diff features, 19K params) | 0.1667 | Skip | t0: 0.2590, t1: 0.0743. Conv2D underperformed - GRU-based approaches better for this task. |
| **exp_20** | Ensemble (exp_01 55% + exp_13 45%) | 0.2746 | **0.2858** | t0: 0.4030, t1: 0.1461. **NEW BEST LB!** Weighted average of exp_01 (GRU) and exp_13 (Hybrid). |
| **exp_21** | Multi-Stream GRU + Temporal Attention (399K params) | 0.2009 | Skip | t0: 0.3032, t1: 0.0985. Causal mask fixed. Worse than exp_20 ensemble - skip LB submission. |
| **exp_22** | 2D Reshaped Temporal Conv (100x32 → 10x10x32, 1.36M params) | 0.0266 | Skip | t0: -0.04, t1: 0.09. Failed - 2D spatial structure doesn't capture temporal dynamics. |
| **exp_23** | TCN+GRU + Enhanced Features (micro-price, OFI, WPP, 285K params) | **0.2679** | 0.2462 | t0: 0.3385, t1: 0.1267. 47 features. **PROMISING - revisit for improvements!** Good t0, Best t1! generalizes well. |
| **exp_24** | LiT-inspired Transformer + GRU (Patch Embed, Causal Attn, 291K params) | **0.1773** | 0.1476 | Train-val: 0.3047 (leaky norm). Full-val: t0: 0.2575, t1: 0.0169. Normalization leakage: train uses full-seq z-score, inference uses online EMA. |
| **exp_27** | LOB-ConvNeXt Tiny + GRU (Causal stem/downsample, no linear interp, ~12.3M params) | 0.1800 | Pending | Epoch 10/50 stop. t0: 0.1788, t1: 0.1824. Causal fixes applied; realistic validation after leakage cleanup. |
| **exp_28** | Advanced Transformer (aicheye/wundernn arch, 135K params) | **0.3001** | Pending | t0: 0.3839, t1: 0.1325. Feature attention, temporal conv, ensemble heads, causal EMA norm. Context=100. Epoch 23/30 best. |