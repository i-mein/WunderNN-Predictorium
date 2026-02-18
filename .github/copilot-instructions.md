# WunderNN-Predictorium AI Agent Instructions

## Project Overview
This is a **Limit Order Book (LOB) prediction competition** (Wunder Challenge 2). The goal is to predict two price movement indicators (`t0`, `t1`) using 32 raw LOB features. Submissions are evaluated using **Weighted Pearson Correlation** with final score = `2/3 * t0_corr + 1/3 * t1_corr`.

**Current best**: exp_20 ensemble (LB 0.2858). Target: 0.324 (leaderboard leader).

## Architecture & Data Flow

### Input Data Structure (32 raw features)
- `p0-p5`: Bid prices (best to worst), `p6-p11`: Ask prices
- `v0-v5`: Bid volumes, `v6-v11`: Ask volumes  
- `dp0-dp3`: Delta prices, `dv0-dv3`: Delta volumes
- Targets: `t0` (short-term), `t1` (long-term price movement)

### Directory Structure
- **`solutions/exp_XX/`**: Self-contained experiment folders with `solution.py`, `src/`, `best_model.pth`
- **`src/utils.py`**: Core utilities (`DataPoint`, `ScorerStepByStep`, `weighted_pearson_correlation`)
- **`datasets/`**: `train.parquet`, `valid.parquet`

## Critical Development Patterns

### Submission Format
Every `solution.py` must define:
```python
class PredictionModel:
    def __init__(self): ...  # Load model weights
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if not data_point.need_prediction:
            return None
        return np.array([t0_pred, t1_pred])  # Shape (2,)
```

### Stateful Inference (Critical)
Models process **one timestep at a time** with hidden state carried across the sequence:
```python
if self.current_seq_ix != data_point.seq_ix:
    self.current_seq_ix = data_point.seq_ix
    self.hidden = None  # Reset on new sequence
output, self.hidden = self.model(input_tensor, self.hidden)
```

### Leakage Prevention (Causal Constraints)
- **All convolutions must use causal (left) padding**: `F.pad(x, (padding, 0))`
- **No bidirectional RNNs** - future information causes validation/LB gap
- **No symmetric padding in CNN** - exp_10 showed 0.59 Val → 0.10 LB with symmetric padding

### Feature Engineering Safety
Use safe math to prevent NaN propagation:
```python
def safe_divide(a, b, default=0.0):
    return np.where(np.abs(b) > 1e-10, a / b, default)
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
```

## Commands & Workflows

### Training
```bash
uv run python solutions/exp_XX/src/train.py
```

### Local Evaluation
```bash
uv run python -c "
from src.utils import ScorerStepByStep
from solutions.exp_XX.solution import PredictionModel
scorer = ScorerStepByStep('datasets/valid.parquet')
print(scorer.score(PredictionModel()))
"
```

### Create Submission
```bash
cd solutions/exp_XX && zip -r ../../submission_exp_XX.zip . && cd ../..
```

## Model Architecture Insights (from experiments)

| Pattern | Result | Recommendation |
|---------|--------|----------------|
| Unidirectional GRU | exp_01: LB 0.282 | ✅ Strong baseline |
| Transformer-GRU hybrid | exp_13: LB 0.226 | ✅ Good for ensembling |
| Complex dual-head | exp_07: TIMEOUT | ⚠️ Avoid pandas in inference |
| Symmetric CNN padding | exp_10: LB 0.10 | ❌ Causes leakage |
| 2D spatial convolutions | exp_22: Val 0.026 | ❌ Doesn't capture temporal dynamics |

## Key Files to Reference
- **Best baseline**: `solutions/exp_01/solution.py` (clean GRU implementation)
- **Feature engineering**: `solutions/exp_23/src/features.py` (safe compute patterns)
- **Evaluation metric**: `src/utils.py::weighted_pearson_correlation`
- **Experiment log**: `artifacts/experiment_log.md`
