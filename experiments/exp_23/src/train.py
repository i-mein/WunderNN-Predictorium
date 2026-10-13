"""
exp_23: Training script for TCN+GRU with Enhanced Features

Key improvements:
- Enhanced LOB features (micro-price, OFI, WPP)
- TCN for local pattern extraction
- GRU for temporal modeling
- Direct Pearson loss option
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from model import TCNGRUModel
from features import compute_features_batch

# ============================================
# Configuration
# ============================================
CONFIG = {
    "experiment": "exp_23_tcn_gru_enhanced",
    "input_dim": 47,      # 32 raw + 15 enhanced
    "tcn_hidden": 64,
    "tcn_layers": 4,
    "gru_hidden": 128,
    "gru_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 30, #15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_steps": 99,   # First 99 steps not scored
    "use_pearson_loss": False,  # Set True to optimize Pearson directly
    "use_wandb": True,           # Set True to enable Weights & Biases logging
    "wandb_project": "WunderNN-Predictorium",
}

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")


# ============================================
# Loss Functions
# ============================================
class PearsonLoss(nn.Module):
    """Loss based on negative Pearson correlation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Flatten for correlation calculation
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Center the predictions and targets
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        
        # Calculate correlation
        cov = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        correlation = cov / (pred_std * target_std)
        
        # Return negative correlation as loss (we want to maximize correlation)
        return 1 - correlation


class WeightedMSELoss(nn.Module):
    """MSE loss weighted by target magnitude (matching competition metric)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        weights = torch.abs(target) + 1e-8
        squared_errors = (pred - target) ** 2
        weighted_loss = (weights * squared_errors).sum() / weights.sum()
        return weighted_loss


# ============================================
# Dataset
# ============================================
class LOBDataset(Dataset):
    """LOB Dataset with enhanced feature computation."""
    
    def __init__(self, parquet_path):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        
        # Get sequence IDs
        self.seq_ids = self.df['seq_ix'].unique()
        print(f"Found {len(self.seq_ids)} sequences")
        
        # Feature and target columns
        self.raw_cols = [c for c in self.df.columns 
                        if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        # Cache data with enhanced features
        print("Computing enhanced features and caching sequences...")
        self.data_cache = {}
        
        grouped = self.df.groupby('seq_ix')
        for seq_ix in tqdm(self.seq_ids, desc="Extracting sequences"):
            group = grouped.get_group(seq_ix)
            
            # Raw features (32,)
            raw_features = group[self.raw_cols].values.astype(np.float32)
            
            # Compute enhanced features for full sequence
            # raw_features: (seq_len, 32) -> enhanced: (seq_len, 47)
            enhanced = np.zeros((raw_features.shape[0], 47), dtype=np.float32)
            for t in range(raw_features.shape[0]):
                enhanced[t] = self._compute_enhanced(raw_features[t])
            
            targets = group[self.target_cols].values.astype(np.float32)
            
            self.data_cache[seq_ix] = (enhanced, targets)
    
    def _compute_enhanced(self, state):
        """Compute enhanced features for a single timestep."""
        from features import compute_enhanced_features
        return compute_enhanced_features(state)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets = self.data_cache[seq_ix]
        return torch.from_numpy(features), torch.from_numpy(targets)


# ============================================
# Metrics
# ============================================
def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Pearson correlation (competition metric)."""
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)
    
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0
    
    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w
    
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred
    
    cov = np.sum(weights * dev_true * dev_pred) / sum_w
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w
    
    if var_true <= 0 or var_pred <= 0:
        return 0.0
    
    return float(cov / (np.sqrt(var_true) * np.sqrt(var_pred)))


# ============================================
# Training
# ============================================
def train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
    
    for X, y in pbar:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(X)
        
        # Loss only on scored steps (99+)
        warmup = config['warmup_steps']
        pred_scored = outputs[:, warmup:, :]
        target_scored = y[:, warmup:, :]
        
        loss = criterion(pred_scored, target_scored)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Collect for metrics
        all_preds.append(pred_scored.detach().cpu().numpy())
        all_targets.append(target_scored.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate training correlations
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    corr_t0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    corr_t1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    
    return total_loss / len(train_loader), corr_t0, corr_t1


def validate(model, val_loader, config):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validating"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            outputs, _ = model(X)
            
            # Score only steps 99+
            warmup = config['warmup_steps']
            pred_scored = outputs[:, warmup:, :]
            target_scored = y[:, warmup:, :]
            
            loss = criterion(pred_scored, target_scored)
            total_loss += loss.item()
            
            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target_scored.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    corr_t0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    corr_t1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    corr_avg = (2 * corr_t0 + corr_t1) / 3  # Weighted average (2/3 t0, 1/3 t1)
    
    return total_loss / len(val_loader), corr_t0, corr_t1, corr_avg


def main():
    config = CONFIG
    
    # Initialize wandb
    wandb.init(project=config["wandb_project"], name=config["experiment"], config=config)
    
    # Data paths
    train_path = os.path.join(PROJECT_ROOT, "datasets", "train.parquet")
    valid_path = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    
    # Load datasets
    train_dataset = LOBDataset(train_path)
    valid_dataset = LOBDataset(valid_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config["batch_size"] * 2, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training: {len(train_dataset)} sequences")
    print(f"Validation: {len(valid_dataset)} sequences")
    
    # Initialize model
    model = TCNGRUModel(
        input_dim=config["input_dim"],
        tcn_hidden=config["tcn_hidden"],
        tcn_layers=config["tcn_layers"],
        gru_hidden=config["gru_hidden"],
        gru_layers=config["gru_layers"],
        dropout=config["dropout"]
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function
    if config["use_pearson_loss"]:
        criterion = PearsonLoss()
    else:
        criterion = WeightedMSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    total_steps = len(train_loader) * config["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    best_val_corr = -1.0
    save_path = os.path.join(os.path.dirname(SCRIPT_DIR), "best_model.pth")
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_t0, train_t1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, epoch, config
        )
        
        # Validate
        val_loss, val_t0, val_t1, val_avg = validate(model, valid_loader, config)
        
        # Log
        print(f"Train - Loss: {train_loss:.4f}, Corr T0: {train_t0:.4f}, Corr T1: {train_t1:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Corr T0: {val_t0:.4f}, Corr T1: {val_t1:.4f}, Avg: {val_avg:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/pearson_t0": train_t0,
            "train/pearson_t1": train_t1,
            "valid/loss": val_loss,
            "valid/pearson_t0": val_t0,
            "valid/pearson_t1": val_t1,
            "valid/pearson_avg": val_avg
        })
        
        # Save best model
        if val_avg > best_val_corr:
            best_val_corr = val_avg
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! Corr avg: {val_avg:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training complete. Best validation correlation: {best_val_corr:.4f}")
    print(f"Model saved to: {save_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
