"""
exp_28: Advanced Transformer for LOB Prediction

Architecture from aicheye/wundernn adapted for:
- 2 regression targets (t0, t1)
- Weighted Pearson correlation metric
- Causal normalization
"""

from .model import AdvancedTransformerPredictor
from .config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from .utils import DataPoint, weighted_pearson_correlation
