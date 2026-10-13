"""
Evaluate ensemble of exp_27, exp_20, exp_23 using full validation set.
Mimics submission scoring via ScorerStepByStep.
"""

import os
import sys
import numpy as np

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from src.utils import ScorerStepByStep, DataPoint
from solutions.exp_27.solution import PredictionModel as Exp27Model
from solutions.exp_20.solution import PredictionModel as Exp20Model
from solutions.exp_23.solution import PredictionModel as Exp23Model

# Per-target weights (will be normalized across available models)
WEIGHTS_T0 = {
    "exp_27": 0.05,  # requested: low weight for t0
    "exp_20": 0.55,
    "exp_23": 0.40,
}

WEIGHTS_T1 = {
    "exp_27": 0.80,  # requested: high weight for t1
    "exp_20": 0.10,
    "exp_23": 0.10,
}


class EnsemblePredictionModel:
    def __init__(self):
        self.models = {
            "exp_27": Exp27Model(),
            "exp_20": Exp20Model(),
            "exp_23": Exp23Model(),
        }

    def _normalize_weights(self, weights, available_names):
        total = sum(weights[name] for name in available_names)
        if total <= 0:
            # Fallback to uniform weights
            return {name: 1.0 / len(available_names) for name in available_names}
        return {name: weights[name] / total for name in available_names}

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if not data_point.need_prediction:
            # Ensure all models still advance their internal state
            for model in self.models.values():
                _ = model.predict(data_point)
            return None

        preds = {}
        for name, model in self.models.items():
            pred = model.predict(data_point)
            if pred is None:
                return None
            preds[name] = pred

        available = list(preds.keys())
        w0 = self._normalize_weights(WEIGHTS_T0, available)
        w1 = self._normalize_weights(WEIGHTS_T1, available)

        pred_t0 = sum(w0[name] * preds[name][0] for name in available)
        pred_t1 = sum(w1[name] * preds[name][1] for name in available)

        return np.array([pred_t0, pred_t1], dtype=np.float32)


def main():
    scorer = ScorerStepByStep("datasets/valid.parquet")
    model = EnsemblePredictionModel()
    results = scorer.score(model)

    print("\n=== Ensemble Results (Full Validation) ===")
    print(f"t0: {results['t0']:.4f}")
    print(f"t1: {results['t1']:.4f}")
    print(f"weighted: {results['weighted_pearson']:.4f}")


if __name__ == "__main__":
    main()
