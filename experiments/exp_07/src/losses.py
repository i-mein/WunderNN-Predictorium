import torch
import torch.nn as nn
import torch.nn.functional as F

def pearson_correlation_loss(y_pred, y_true, weights=None):
    """
    Negative Pearson correlation as loss (to maximize correlation).
    Optionally weighted by target amplitude.
    
    Args:
        y_pred: Predictions (batch, seq_len) or (batch*seq_len,)
        y_true: Targets (batch, seq_len) or (batch*seq_len,)
        weights: Optional weights (batch, seq_len) or (batch*seq_len,)
    
    Returns:
        Negative Pearson correlation (scalar)
    """
    # Flatten if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Use absolute value of targets as weights if not provided
    if weights is None:
        weights = torch.abs(y_true)
        weights = torch.clamp(weights, min=1e-8)
    else:
        weights = weights.reshape(-1)
    
    # Weighted means
    sum_w = weights.sum()
    if sum_w == 0:
        return torch.tensor(0.0, device=y_pred.device)
    
    mean_pred = (y_pred * weights).sum() / sum_w
    mean_true = (y_true * weights).sum() / sum_w
    
    # Centered values
    pred_centered = y_pred - mean_pred
    true_centered = y_true - mean_true
    
    # Weighted covariance and variances
    cov = (weights * pred_centered * true_centered).sum() / sum_w
    var_pred = (weights * pred_centered ** 2).sum() / sum_w
    var_true = (weights * true_centered ** 2).sum() / sum_w
    
    # Pearson correlation
    corr = cov / (torch.sqrt(var_pred * var_true) + 1e-8)
    
    # Return negative (to minimize)
    return -corr


def combined_loss(pred_t0, pred_t1, target_t0, target_t1, 
                  weight_t0=0.7, weight_t1=0.3, mse_reg_weight=0.1):
    """
    Combined loss for dual-head model.
    
    Args:
        pred_t0: Predictions for t0 (batch, seq_len, 1)
        pred_t1: Predictions for t1 (batch, seq_len, 1)
        target_t0: Targets for t0 (batch, seq_len, 1)
        target_t1: Targets for t1 (batch, seq_len, 1)
        weight_t0: Weight for t0 loss (default: 0.7)
        weight_t1: Weight for t1 loss (default: 0.3)
        mse_reg_weight: MSE regularization weight for t1 (default: 0.1)
    
    Returns:
        total_loss: Combined loss
        loss_t0: t0 loss (for logging)
        loss_t1: t1 loss (for logging)
    """
    # Squeeze last dimension
    pred_t0 = pred_t0.squeeze(-1)
    pred_t1 = pred_t1.squeeze(-1)
    target_t0 = target_t0.squeeze(-1)
    target_t1 = target_t1.squeeze(-1)
    
    # t0 loss (strong predictor) - Pearson correlation only
    loss_t0 = pearson_correlation_loss(pred_t0, target_t0)
    
    # t1 loss (weak predictor) - Pearson + MSE regularization
    loss_t1_pearson = pearson_correlation_loss(pred_t1, target_t1)
    loss_t1_mse = F.mse_loss(pred_t1, target_t1)
    loss_t1 = loss_t1_pearson + mse_reg_weight * loss_t1_mse
    
    # Combined loss
    total_loss = weight_t0 * loss_t0 + weight_t1 * loss_t1
    
    return total_loss, loss_t0, loss_t1
