"""exp_29: LOBViT 2D Transformer utilities."""

import numpy as np
from dataclasses import dataclass


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Pearson correlation (competition metric)."""
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)
    weights = np.maximum(np.abs(y_true), 1e-8)
    
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0
    
    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w
    
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred
    
    cov = np.sum(weights * dev_true * dev_pred) / sum_w
    var_true = np.sum(weights * dev_true ** 2) / sum_w
    var_pred = np.sum(weights * dev_pred ** 2) / sum_w
    
    if var_true <= 0 or var_pred <= 0:
        return 0.0
    
    return float(cov / (np.sqrt(var_true) * np.sqrt(var_pred)))
