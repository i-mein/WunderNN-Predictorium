"""
Loss functions for exp_09.

Pearson correlation loss for direct metric optimization.
"""
import torch
import torch.nn.functional as F


def pearson_correlation_loss(y_pred, y_true, eps=1e-8):
    """
    Negative Pearson correlation as loss (to maximize correlation).
    
    Args:
        y_pred: Predictions (batch, seq_len) or (N,)
        y_true: Targets (batch, seq_len) or (N,)
        eps: Small constant for numerical stability
    
    Returns:
        Negative Pearson correlation (scalar)
    """
    # Flatten if needed
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Mean center
    pred_mean = y_pred.mean()
    true_mean = y_true.mean()
    
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    
    # Covariance
    cov = (pred_centered * true_centered).mean()
    
    # Standard deviations
    pred_std = torch.sqrt((pred_centered ** 2).mean() + eps)
    true_std = torch.sqrt((true_centered ** 2).mean() + eps)
    
    # Pearson correlation
    corr = cov / (pred_std * true_std)
    
    # Return negative (to minimize)
    return -corr


def weighted_pearson_loss(y_pred, y_true, weights=None, eps=1e-8):
    """
    Weighted Pearson correlation loss.
    
    Weights can emphasize certain samples (e.g., larger target values).
    
    Args:
        y_pred: Predictions (N,)
        y_true: Targets (N,)
        weights: Optional weights (N,). If None, uses |y_true| as weights.
        eps: Small constant for numerical stability
    
    Returns:
        Negative weighted Pearson correlation (scalar)
    """
    # Flatten
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Default weights based on target magnitude
    if weights is None:
        weights = torch.abs(y_true) + eps
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted means
    pred_mean = (y_pred * weights).sum()
    true_mean = (y_true * weights).sum()
    
    # Centered values
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    
    # Weighted covariance and variances
    cov = (weights * pred_centered * true_centered).sum()
    var_pred = (weights * pred_centered ** 2).sum()
    var_true = (weights * true_centered ** 2).sum()
    
    # Pearson correlation
    corr = cov / (torch.sqrt(var_pred * var_true) + eps)
    
    return -corr


def combined_loss(pred_t0, pred_t1, target_t0, target_t1,
                  weight_t0=0.5, weight_t1=0.5, mse_reg=0.1, use_mse_only=False):
    """
    Combined loss for dual-head model.
    
    Args:
        pred_t0: Predictions for t0
        pred_t1: Predictions for t1
        target_t0: Targets for t0
        target_t1: Targets for t1
        weight_t0: Weight for t0 loss
        weight_t1: Weight for t1 loss
        mse_reg: MSE regularization weight (for stability)
        use_mse_only: If True, use only MSE loss (no Pearson)
    
    Returns:
        total_loss: Combined loss
        loss_t0: t0 component
        loss_t1: t1 component
    """
    # MSE loss for each target
    loss_t0_mse = F.mse_loss(pred_t0.view(-1), target_t0.view(-1))
    loss_t1_mse = F.mse_loss(pred_t1.view(-1), target_t1.view(-1))
    
    if use_mse_only:
        # Pure MSE loss (like exp_06)
        loss_t0 = loss_t0_mse
        loss_t1 = loss_t1_mse
    else:
        # Pearson correlation loss for each target
        loss_t0_pearson = pearson_correlation_loss(pred_t0, target_t0)
        loss_t1_pearson = pearson_correlation_loss(pred_t1, target_t1)
        
        # Combined per-target losses
        loss_t0 = loss_t0_pearson + mse_reg * loss_t0_mse
        loss_t1 = loss_t1_pearson + mse_reg * loss_t1_mse
    
    # Total loss
    total_loss = weight_t0 * loss_t0 + weight_t1 * loss_t1
    
    return total_loss, loss_t0, loss_t1
