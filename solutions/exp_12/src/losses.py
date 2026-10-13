"""
Loss functions for exp_10: DeepLOB-style training

Key design choices:
1. Huber loss (robust to outliers, better than MSE)
2. Weighted combination for t0/t1
3. Optional Pearson correlation component
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLossWeighted(nn.Module):
    """
    Weighted Huber loss for dual-target prediction.
    
    Huber loss is more robust to outliers than MSE:
    - Quadratic for small errors (like MSE)
    - Linear for large errors (like L1)
    
    Args:
        t0_weight: Weight for t0 loss (default 0.6)
        t1_weight: Weight for t1 loss (default 0.4)
        delta: Huber delta threshold (default 1.0)
    """
    
    def __init__(self, t0_weight=0.6, t1_weight=0.4, delta=1.0):
        super().__init__()
        self.t0_weight = t0_weight
        self.t1_weight = t1_weight
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
    
    def forward(self, pred_t0, pred_t1, target_t0, target_t1, mask):
        """
        Compute weighted Huber loss.
        
        Args:
            pred_t0: Predictions for t0 (batch, seq, 1)
            pred_t1: Predictions for t1 (batch, seq, 1)
            target_t0: Targets for t0 (batch, seq, 1)
            target_t1: Targets for t1 (batch, seq, 1)
            mask: Valid sample mask (batch, seq, 1) - True where need_prediction=1
        
        Returns:
            total_loss: Weighted sum of t0 and t1 losses
            loss_t0: Individual t0 loss
            loss_t1: Individual t1 loss
        """
        # Compute losses
        loss_t0_all = self.huber(pred_t0, target_t0)
        loss_t1_all = self.huber(pred_t1, target_t1)
        
        # Apply mask and average
        if mask is not None:
            mask_float = mask.float()
            n_valid = mask_float.sum() + 1e-8
            
            loss_t0 = (loss_t0_all * mask_float).sum() / n_valid
            loss_t1 = (loss_t1_all * mask_float).sum() / n_valid
        else:
            loss_t0 = loss_t0_all.mean()
            loss_t1 = loss_t1_all.mean()
        
        # Weighted combination
        total_loss = self.t0_weight * loss_t0 + self.t1_weight * loss_t1
        
        return total_loss, loss_t0, loss_t1


class HuberPearsonLoss(nn.Module):
    """
    Combined Huber + Pearson correlation loss.
    
    Pearson correlation directly optimizes our evaluation metric,
    but can be unstable. Huber provides a stable gradient.
    
    L = alpha * Huber + (1 - alpha) * (1 - Pearson)
    
    Args:
        alpha: Weight for Huber component (default 0.5)
        t0_weight: Weight for t0 loss (default 0.6)
        t1_weight: Weight for t1 loss (default 0.4)
        delta: Huber delta threshold (default 1.0)
    """
    
    def __init__(self, alpha=0.5, t0_weight=0.6, t1_weight=0.4, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.t0_weight = t0_weight
        self.t1_weight = t1_weight
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
    
    def pearson_loss(self, pred, target, mask):
        """
        Compute negative Pearson correlation as loss.
        
        Returns 1 - Pearson, so lower is better.
        """
        if mask is not None:
            pred_masked = pred[mask]
            target_masked = target[mask]
        else:
            pred_masked = pred.flatten()
            target_masked = target.flatten()
        
        if len(pred_masked) < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Center
        pred_centered = pred_masked - pred_masked.mean()
        target_centered = target_masked - target_masked.mean()
        
        # Correlation
        cov = (pred_centered * target_centered).sum()
        std_pred = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        std_target = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        pearson = cov / (std_pred * std_target + 1e-8)
        
        # Clamp to valid range
        pearson = torch.clamp(pearson, -1.0, 1.0)
        
        return 1.0 - pearson  # Minimize this
    
    def forward(self, pred_t0, pred_t1, target_t0, target_t1, mask):
        """
        Compute combined loss.
        
        Args:
            pred_t0: Predictions for t0 (batch, seq, 1)
            pred_t1: Predictions for t1 (batch, seq, 1)
            target_t0: Targets for t0 (batch, seq, 1)
            target_t1: Targets for t1 (batch, seq, 1)
            mask: Valid sample mask (batch, seq, 1)
        
        Returns:
            total_loss: Combined loss
            loss_t0: Individual t0 loss
            loss_t1: Individual t1 loss
        """
        # Huber component
        loss_t0_huber = self.huber(pred_t0, target_t0)
        loss_t1_huber = self.huber(pred_t1, target_t1)
        
        if mask is not None:
            mask_float = mask.float()
            n_valid = mask_float.sum() + 1e-8
            
            huber_t0 = (loss_t0_huber * mask_float).sum() / n_valid
            huber_t1 = (loss_t1_huber * mask_float).sum() / n_valid
            
            # Pearson component
            mask_bool = mask.squeeze(-1).bool()
            pearson_t0 = self.pearson_loss(pred_t0.squeeze(-1), target_t0.squeeze(-1), mask_bool)
            pearson_t1 = self.pearson_loss(pred_t1.squeeze(-1), target_t1.squeeze(-1), mask_bool)
        else:
            huber_t0 = loss_t0_huber.mean()
            huber_t1 = loss_t1_huber.mean()
            
            pearson_t0 = self.pearson_loss(pred_t0, target_t0, None)
            pearson_t1 = self.pearson_loss(pred_t1, target_t1, None)
        
        # Combine Huber and Pearson
        loss_t0 = self.alpha * huber_t0 + (1 - self.alpha) * pearson_t0
        loss_t1 = self.alpha * huber_t1 + (1 - self.alpha) * pearson_t1
        
        # Weighted combination
        total_loss = self.t0_weight * loss_t0 + self.t1_weight * loss_t1
        
        return total_loss, loss_t0, loss_t1


def compute_pearson_correlation(pred, target, mask=None):
    """
    Compute Pearson correlation (for evaluation).
    
    Args:
        pred: Predictions (any shape, will be flattened)
        target: Targets (same shape as pred)
        mask: Optional mask (True for valid samples)
    
    Returns:
        Pearson correlation coefficient
    """
    pred = pred.flatten()
    target = target.flatten()
    
    if mask is not None:
        mask = mask.flatten()
        pred = pred[mask]
        target = target[mask]
    
    if len(pred) < 2:
        return 0.0
    
    # Center
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    # Correlation
    cov = (pred_centered * target_centered).sum()
    std_pred = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
    std_target = torch.sqrt((target_centered ** 2).sum() + 1e-8)
    
    pearson = cov / (std_pred * std_target + 1e-8)
    
    return pearson.item()


def test_losses():
    """Test loss functions."""
    batch, seq = 4, 100
    
    pred_t0 = torch.randn(batch, seq, 1)
    pred_t1 = torch.randn(batch, seq, 1)
    target_t0 = torch.randn(batch, seq, 1)
    target_t1 = torch.randn(batch, seq, 1)
    mask = torch.randint(0, 2, (batch, seq, 1)).float()
    
    # Test Huber loss
    huber_loss = HuberLossWeighted()
    total, l_t0, l_t1 = huber_loss(pred_t0, pred_t1, target_t0, target_t1, mask)
    print(f"Huber Loss: total={total:.4f}, t0={l_t0:.4f}, t1={l_t1:.4f}")
    
    # Test Huber + Pearson loss
    huber_pearson = HuberPearsonLoss(alpha=0.5)
    total, l_t0, l_t1 = huber_pearson(pred_t0, pred_t1, target_t0, target_t1, mask)
    print(f"Huber+Pearson Loss: total={total:.4f}, t0={l_t0:.4f}, t1={l_t1:.4f}")
    
    # Test Pearson correlation
    pearson = compute_pearson_correlation(pred_t0, target_t0, mask.bool())
    print(f"Pearson correlation (random): {pearson:.4f}")
    
    # Test with perfect correlation
    perfect_t0 = target_t0.clone()
    pearson_perfect = compute_pearson_correlation(perfect_t0, target_t0, mask.bool())
    print(f"Pearson correlation (perfect): {pearson_perfect:.4f}")


if __name__ == "__main__":
    test_losses()
