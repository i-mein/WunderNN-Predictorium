# Walkthrough - Experiment 01: specific PyTorch GRU

## Solution Overview
We implemented a **Bidirectional GRU** (modified to Unidirectional for causality) model using PyTorch to predict future price movements (`t0`, `t1`) from limit order book sequences.

## Solution Location
- **Code**: `solutions/exp_01/solution.py`
- **Model Weights**: `solutions/exp_01/best_model.pth`
- **Training Script**: `src/train.py`
- **Model Definition**: `src/model.py`

## Architecture
- **Input**: 32 features (LOB data) per time step.
- **Backbone**: 2-layer GRU with 256 hidden units.
- **Head**: Fully Connected layer (256 -> 64 -> 2) with ReLU and Dropout (0.2).
- **Optimization**: AdamW optimizer, OneCycleLR scheduler, MSE Loss.

## Results

| Metric | Validation | Leaderboard |
| :--- | :--- | :--- |
| **Mean Weighted Pearson** | **0.2310** | **0.2820** |
| t0 Correlation | 0.3342 | - |
| t1 Correlation | 0.1277 | - |

> [!NOTE]
> Leaderboard score (0.2820) is significantly higher than validation score (0.2310), suggesting good generalization or distribution shift in the test set favoring our model.

## How to Run

### 1. Training
To retrain the model:
```bash
# From project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
.venv/bin/python src/train.py
```
This will save `best_model.pth` in the project root (or wherever the script runs). Move it to `solutions/exp_01/` for the solution to pick it up.

### 2. Evaluation
To evaluate the solution locally:
```bash
.venv/bin/python solutions/exp_01/solution.py
```

## Key Changes
- Migrated from ONNX Runtime (Baseline) to **PyTorch**.
- Implemented stateful GRU inference in `PredictionModel.predict`.
- Refactored solution to dedicated experiment folder `solutions/exp_01` as requested.
- Restored `solutions/example_solution/solution.py` to its original state.
