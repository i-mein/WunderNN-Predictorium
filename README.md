# WunderNN Predictorium

This repository contains my work for the WunderNN Predictorium limit order book prediction competition. The task was to predict two future price-movement targets, `t0` and `t1`, from sequential market data. The final score was `2/3 * corr(t0) + 1/3 * corr(t1)`, using weighted Pearson correlations.

## Competition result

**54th place out of 5,094 participants** — a finish in approximately the top 1.1%.

See the [official final standings](https://wundernn.io/predictorium/finals).

## Competition work

I developed and compared a series of models and feature sets throughout the competition, moving from a unidirectional GRU baseline to feature-engineered recurrent models, CNN-GRU and Transformer-GRU hybrids, multi-stream architectures, temporal convolutional networks, CatBoost models, and ensembles.

Experiments focused on improving generalization to step-by-step inference, not just offline validation. In particular, I investigated look-ahead leakage caused by non-causal convolutions and full-sequence normalization. That work informed the use of causal attention and features derived only from the current observation, along with explicit recurrent-state handling at sequence boundaries. I also explored order-book signals such as spread, mid-price, volume imbalance, momentum, and book pressure.

## Final submission

The submitted model in [`submission/final/`](submission/final/) is a multi-stream GRU with temporal and cross-stream attention:

- It processes prices, volumes, price deltas, volume deltas, and eight engineered order-book features in separate streams.
- Each stream uses a GRU encoder followed by causal temporal attention; a causal cross-stream attention layer fuses the streams.
- A shared fusion network feeds separate prediction heads for `t0` and `t1`.
- Inference runs on CPU and carries each stream's hidden state forward one timestep at a time, resetting it when the stock or day changes.

The self-contained inference entry point is [`submission/final/solution.py`](submission/final/solution.py), the model definition is in [`submission/final/src/model.py`](submission/final/src/model.py), and the saved weights are [`submission/final/best_model.pth`](submission/final/best_model.pth). The training script is [`submission/final/src/train.py`](submission/final/src/train.py). Experiment notes and supporting artifacts are collected in [`artifacts/`](artifacts/).

## Repository layout

- `submission/final/` — final inference code, model, and checkpoint.
- `experiments/` — competition experiments, alternative submissions, and model artifacts.
- `artifacts/` — experiment log, research notes, and result visualizations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
