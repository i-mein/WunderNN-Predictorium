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

### Normalization Leakage (Critical)
**Training normalization MUST match inference normalization!**

❌ **WRONG** - Full-sequence z-score (sees future):
```python
# In training dataset - LEAKY!
mean = features.mean(axis=0)  # Uses FULL sequence
std = features.std(axis=0)
normalized = (features - mean) / std
```

✅ **CORRECT** - Online/causal normalization (only past):
```python
# In inference - CAUSAL
alpha = 0.01
running_mean = (1 - alpha) * running_mean + alpha * current_state
running_var = (1 - alpha) * running_var + alpha * (current_state - running_mean) ** 2
```

exp_24 showed Train-Val: 0.3047 → Full-Val: 0.1773 due to this leakage!

### Feature Engineering Safety
Use safe math to prevent NaN propagation:
```python
def safe_divide(a, b, default=0.0):
    return np.where(np.abs(b) > 1e-10, a / b, default)
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
```

### Configuration Consistency (Critical)
**Model config MUST be shared between `train.py` and `solution.py`!**

Create `src/config.py` with shared configuration:
```python
# src/config.py
MODEL_CONFIG = {
    "input_dim": 32,
    "hidden_dim": 128,
    "pool_type": "mean",  # Architecture choices here!
    # ... all model hyperparameters
}

TRAIN_CONFIG = {
    "dropout": 0.2,
    "epochs": 10,
    # ... training-specific settings
}

def get_model_kwargs(for_inference=False):
    """Returns model kwargs with dropout=0 for inference."""
    return {**MODEL_CONFIG, "dropout": 0.0 if for_inference else TRAIN_CONFIG["dropout"]}
```

Then import in both files:
```python
# train.py
from config import CONFIG, get_model_kwargs
model = MyModel(**get_model_kwargs(for_inference=False))

# solution.py
from config import MODEL_CONFIG, get_model_kwargs
model = MyModel(**get_model_kwargs(for_inference=True))
```

exp_25b showed architecture mismatch (`pool_type='attention'` vs `'mean'`) causing load failures!

## Commands & Workflows

### Training
```bash
uv run python solutions/exp_XX/src/train.py
```

### Local Evaluation (CRITICAL - Always Run After Training)
**Training validation scores can be MISLEADING due to normalization leakage.**
Always run full validation using the organizer's `ScorerStepByStep` after training.
The scorer processes data point-by-point exactly as the competition does.

Record BOTH scores in experiment log:
- **Train-Val**: Score reported during training (may be leaky)
- **Full-Val**: Score from ScorerStepByStep (true generalization measure)

```bash
uv run python -c "
from src.utils import ScorerStepByStep
from solutions.exp_XX.solution import PredictionModel
scorer = ScorerStepByStep('datasets/valid.parquet')
print(scorer.score(PredictionModel()))
"
```

Or run the solution.py directly if it has a `__main__` block:
```bash
uv run python solutions/exp_XX/solution.py
```

### Create Submission
```bash
cd solutions/exp_XX && zip -r ../../submissions/submission_exp_XX.zip . -x "src/*" -x "__pycache__/*" && cd ../..
```

## Model Architecture Insights (from experiments)

| Pattern | Result | Recommendation |
|---------|--------|----------------|
| Unidirectional GRU | exp_01: LB 0.282 | ✅ Strong baseline |
| Transformer-GRU hybrid | exp_13: LB 0.226 | ✅ Good for ensembling |
| TCN+GRU + Enhanced Features | exp_23: Val 0.268 | ✅ Good t0 (0.34), candidate for ensemble |
| LiT Transformer | exp_24: Full-Val 0.177 | ⚠️ Train-Val inflated (0.30) due to norm leakage |
| Complex dual-head | exp_07: TIMEOUT | ⚠️ Avoid pandas in inference |
| Symmetric CNN padding | exp_10: LB 0.10 | ❌ Causes leakage |
| 2D spatial convolutions | exp_22: Val 0.026 | ❌ Doesn't capture temporal dynamics |

## Key Files to Reference
- **Best baseline**: `solutions/exp_01/solution.py` (clean GRU implementation)
- **Feature engineering**: `solutions/exp_23/src/features.py` (safe compute patterns)
- **Evaluation metric**: `src/utils.py::weighted_pearson_correlation`
- **Experiment log**: `artifacts/experiment_log.md`
