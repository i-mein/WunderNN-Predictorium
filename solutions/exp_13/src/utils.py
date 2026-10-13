"""
Utility functions for exp_13.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DataPoint:
    """Data point for inference."""
    seq_ix: int
    step_in_seq: int
    state: np.ndarray
    need_prediction: bool


def weighted_pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return 0.0
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(np.sum((y_true - mean_true)**2) * np.sum((y_pred - mean_pred)**2))
    
    if denominator < 1e-8:
        return 0.0
    
    return numerator / denominator
