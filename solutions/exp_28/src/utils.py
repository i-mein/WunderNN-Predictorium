"""
exp_28: Utility classes and functions

DataPoint: Data structure for LOB state at each timestep
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DataPoint:
    """
    Data point for a single timestep in LOB sequence.
    
    Attributes:
        seq_ix: Sequence ID
        step_in_seq: Step number within sequence
        need_prediction: Whether prediction is required for this step
        state: numpy array of 32 LOB features
    """
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate weighted Pearson correlation (competition metric).
    
    Weights are based on target magnitude (larger movements weighted more).
    Predictions are clipped to [-6, 6].
    """
    # Clip predictions
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)
    
    # Weights based on target amplitude
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)
    
    # Weighted means
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0
    
    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w
    
    # Weighted deviations
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred
    
    # Weighted covariance
    cov = np.sum(weights * dev_true * dev_pred) / sum_w
    
    # Weighted variances
    var_true = np.sum(weights * dev_true ** 2) / sum_w
    var_pred = np.sum(weights * dev_pred ** 2) / sum_w
    
    # Correlation
    if var_true <= 0 or var_pred <= 0:
        return 0.0
    
    corr = cov / (np.sqrt(var_true) * np.sqrt(var_pred))
    
    return float(corr)
