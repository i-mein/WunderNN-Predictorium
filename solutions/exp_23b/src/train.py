"""
exp_23b: Training script for Multi-Stream TCN + GRU with Attention

Key improvements over exp_23:
- Multi-stream architecture with attention fusion
- Causal normalization matching inference
- First 100 timesteps excluded from loss/metrics (warmup)
- Snapshot ensemble with epoch checkpoints
- Full validation using solution.py at end of training
"""

import os
import sys
import subprocess
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

from model import MultiStreamTCNGRUModel
from features import compute_enhanced_features
from config import CONFIG, TRAIN_CONFIG, get_model_kwargs

# ============================================
# Device Setup
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")


# ============================================
# Loss Functions
# ============================================
class WeightedMSELoss(nn.Module):
    """MSE loss weighted by target magnitude."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        weights = torch.abs(target) + 1e-8
        squared_errors = (pred - target) ** 2
        weighted_loss = (weights * squared_errors).sum() / weights.sum()
        return weighted_loss


class CombinedLoss(nn.Module):
    """Combine MSE and Pearson correlation loss."""
    
    def __init__(self, mse_weight=0.7, pearson_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.mse = WeightedMSELoss()
    
    def pearson_loss(self, pred, target):
        # t0 correlation
        p0 = pred[:, 0] - pred[:, 0].mean()
        t0 = target[:, 0] - target[:, 0].mean()
        corr_t0 = (p0 * t0).sum() / (torch.sqrt((p0**2).sum() * (t0**2).sum()) + 1e-8)
        
        # t1 correlation
        p1 = pred[:, 1] - pred[:, 1].mean()
        t1 = target[:, 1] - target[:, 1].mean()
        corr_t1 = (p1 * t1).sum() / (torch.sqrt((p1**2).sum() * (t1**2).sum()) + 1e-8)
        
        # Weighted (matching competition metric)
        weighted_corr = (2/3) * corr_t0 + (1/3) * corr_t1
        return 1 - weighted_corr
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pearson_loss = self.pearson_loss(pred, target)
        return self.mse_weight * mse_loss + self.pearson_weight * pearson_loss


# ============================================
# Dataset with Causal Normalization
# ============================================
class LOBDataset(Dataset):
    """LOB Dataset with causal normalization and enhanced features."""
    
    def __init__(self, parquet_path, norm_alpha=0.01):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.norm_alpha = norm_alpha
        
        # Get sequence IDs
        self.seq_ids = self.df['seq_ix'].unique()
        print(f"Found {len(self.seq_ids)} sequences")
        
        # Feature and target columns
        self.raw_cols = [c for c in self.df.columns 
                        if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        # Cache data with enhanced features and causal normalization
        print("Computing enhanced features with causal normalization...")
        self.data_cache = {}
        
        grouped = self.df.groupby('seq_ix')
        for seq_ix in tqdm(self.seq_ids, desc="Processing sequences"):
            group = grouped.get_group(seq_ix)
            
            # Raw features
            raw_features = group[self.raw_cols].values.astype(np.float32)
            
            # Compute enhanced features
            enhanced = np.zeros((raw_features.shape[0], 47), dtype=np.float32)
            for t in range(raw_features.shape[0]):
                enhanced[t] = compute_enhanced_features(raw_features[t])
            
            # Apply causal normalization
            normalized = self._normalize_causal(enhanced)
            
            targets = group[self.target_cols].values.astype(np.float32)
            need_pred = group['need_prediction'].values.astype(bool)
            
            self.data_cache[seq_ix] = {
                'features': normalized,
                'targets': targets,
                'need_pred': need_pred
            }
        
        print(f"Cached {len(self.data_cache)} sequences")
    
    def _normalize_causal(self, features):
        """
        Causal (online) normalization using EMA.
        Matches inference behavior exactly.
        """
        seq_len, num_features = features.shape
        normalized = np.zeros_like(features)
        
        running_mean = features[0].copy()
        running_var = np.ones(num_features, dtype=np.float32)
        
        alpha = self.norm_alpha
        
        for t in range(seq_len):
            x = features[t]
            
            # Normalize using current running statistics
            std = np.sqrt(running_var + 1e-8)
            normalized[t] = (x - running_mean) / std
            
            # Update running statistics
            delta = x - running_mean
            running_mean = running_mean + alpha * delta
            running_var = (1 - alpha) * running_var + alpha * (delta ** 2)
        
        return normalized
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        data = self.data_cache[seq_ix]
        
        return {
            'features': torch.tensor(data['features'], dtype=torch.float32),
            'targets': torch.tensor(data['targets'], dtype=torch.float32),
            'need_pred': torch.tensor(data['need_pred'], dtype=torch.bool)
        }


def collate_fn(batch):
    """Collate variable-length sequences with padding."""
    max_len = max(item['features'].shape[0] for item in batch)
    
    features_list = []
    targets_list = []
    need_pred_list = []
    lengths = []
    
    for item in batch:
        seq_len = item['features'].shape[0]
        lengths.append(seq_len)
        
        # Pad features
        feat_pad = torch.zeros(max_len, item['features'].shape[1])
        feat_pad[:seq_len] = item['features']
        features_list.append(feat_pad)
        
        # Pad targets
        tgt_pad = torch.zeros(max_len, 2)
        tgt_pad[:seq_len] = item['targets']
        targets_list.append(tgt_pad)
        
        # Pad need_pred
        np_pad = torch.zeros(max_len, dtype=torch.bool)
        np_pad[:seq_len] = item['need_pred']
        need_pred_list.append(np_pad)
    
    return {
        'features': torch.stack(features_list),
        'targets': torch.stack(targets_list),
        'need_pred': torch.stack(need_pred_list),
        'lengths': torch.tensor(lengths)
    }


# ============================================
# Metrics
# ============================================
def weighted_pearson_correlation(pred_t0, true_t0, pred_t1, true_t1):
    """Calculate weighted Pearson correlation (competition metric)."""
    pred_t0 = np.array(pred_t0)
    true_t0 = np.array(true_t0)
    pred_t1 = np.array(pred_t1)
    true_t1 = np.array(true_t1)
    
    def pearson(pred, true):
        if len(pred) == 0:
            return 0.0
        pred_centered = pred - np.mean(pred)
        true_centered = true - np.mean(true)
        num = np.sum(pred_centered * true_centered)
        denom = np.sqrt(np.sum(pred_centered**2) * np.sum(true_centered**2) + 1e-10)
        return num / denom
    
    t0_corr = pearson(pred_t0, true_t0)
    t1_corr = pearson(pred_t1, true_t1)
    weighted = (2/3) * t0_corr + (1/3) * t1_corr
    
    return {
        't0_corr': t0_corr,
        't1_corr': t1_corr,
        'weighted_score': weighted
    }


# ============================================
# Training
# ============================================
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    all_pred_t0, all_true_t0 = [], []
    all_pred_t1, all_true_t1 = [], []
    
    warmup = CONFIG["warmup_steps"]
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        need_pred = batch['need_pred'].to(device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(features)
        
        # Mask: only scored timesteps (after warmup AND need_pred=True)
        batch_size, seq_len, _ = predictions.shape
        
        # Create warmup mask (exclude first 100 timesteps)
        warmup_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            warmup_mask[i, warmup:lengths[i].item()] = True
        
        # Combine with need_pred
        mask = warmup_mask & need_pred
        
        # Flatten and apply mask
        pred_flat = predictions.view(-1, 2)[mask.view(-1)]
        target_flat = targets.view(-1, 2)[mask.view(-1)]
        
        if pred_flat.shape[0] == 0:
            continue
        
        # Loss
        loss = criterion(pred_flat, target_flat)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Collect predictions for metrics
        all_pred_t0.extend(pred_flat[:, 0].detach().cpu().numpy())
        all_true_t0.extend(target_flat[:, 0].detach().cpu().numpy())
        all_pred_t1.extend(pred_flat[:, 1].detach().cpu().numpy())
        all_true_t1.extend(target_flat[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    metrics = weighted_pearson_correlation(all_pred_t0, all_true_t0, all_pred_t1, all_true_t1)
    
    return total_loss / max(num_batches, 1), metrics


def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    all_pred_t0, all_true_t0 = [], []
    all_pred_t1, all_true_t1 = [], []
    
    warmup = CONFIG["warmup_steps"]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            need_pred = batch['need_pred'].to(device)
            lengths = batch['lengths']
            
            # Forward pass
            predictions, _ = model(features)
            
            # Extract predictions (after warmup AND need_pred=True)
            batch_size = features.shape[0]
            for i in range(batch_size):
                seq_len = lengths[i].item()
                for t in range(warmup, seq_len):
                    if need_pred[i, t]:
                        all_pred_t0.append(predictions[i, t, 0].cpu().item())
                        all_true_t0.append(targets[i, t, 0].cpu().item())
                        all_pred_t1.append(predictions[i, t, 1].cpu().item())
                        all_true_t1.append(targets[i, t, 1].cpu().item())
    
    return weighted_pearson_correlation(all_pred_t0, all_true_t0, all_pred_t1, all_true_t1)


# ============================================
# Main
# ============================================
def main():
    print("=" * 60)
    print("exp_23b: Multi-Stream TCN + GRU with Attention")
    print("=" * 60)
    
    if CONFIG["use_wandb"]:
        wandb.init(project=CONFIG["wandb_project"], name=CONFIG["experiment"], config=CONFIG)
    
    # Load datasets
    train_path = os.path.join(PROJECT_ROOT, "datasets", "train.parquet")
    valid_path = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    
    norm_alpha = CONFIG.get("norm_ema_alpha", 0.01)
    train_dataset = LOBDataset(train_path, norm_alpha=norm_alpha)
    valid_dataset = LOBDataset(valid_path, norm_alpha=norm_alpha)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    model = MultiStreamTCNGRUModel(**get_model_kwargs(for_inference=False)).to(DEVICE)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        mse_weight=CONFIG.get("mse_weight", 0.7),
        pearson_weight=CONFIG.get("pearson_weight", 0.3)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader), T_mult=1
    )
    
    # Snapshot tracking
    num_snapshots = CONFIG.get("num_snapshots", 4)
    best_checkpoints = []  # (score, epoch, path)
    best_score = -float('inf')
    
    base_dir = os.path.dirname(SCRIPT_DIR)
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler)
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Train t0_corr: {train_metrics['t0_corr']:.4f}")
        print(f"Train t1_corr: {train_metrics['t1_corr']:.4f}")
        print(f"Train Weighted: {train_metrics['weighted_score']:.4f}")
        
        # Evaluate
        val_metrics = evaluate_model(model, valid_loader, DEVICE)
        print(f"Val t0_corr: {val_metrics['t0_corr']:.4f}")
        print(f"Val t1_corr: {val_metrics['t1_corr']:.4f}")
        print(f"Val Weighted Score: {val_metrics['weighted_score']:.4f}")
        
        # Log to wandb
        if CONFIG["use_wandb"]:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_t0_corr': train_metrics['t0_corr'],
                'train_t1_corr': train_metrics['t1_corr'],
                'train_weighted': train_metrics['weighted_score'],
                'val_t0_corr': val_metrics['t0_corr'],
                'val_t1_corr': val_metrics['t1_corr'],
                'val_weighted_score': val_metrics['weighted_score'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save snapshot
        current_score = val_metrics['weighted_score']
        snapshot_path = os.path.join(base_dir, f"best_model_ep_{epoch}.pth")
        torch.save(model.state_dict(), snapshot_path)
        print(f"Saved snapshot: best_model_ep_{epoch}.pth")
        
        # Track best checkpoints
        best_checkpoints.append((current_score, epoch, snapshot_path))
        best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top N snapshots
        if len(best_checkpoints) > num_snapshots:
            _, _, path_to_remove = best_checkpoints.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                print(f"Removed old snapshot: {os.path.basename(path_to_remove)}")
        
        # Track best overall
        if current_score > best_score:
            best_score = current_score
            best_path = os.path.join(base_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model! Score: {best_score:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation score (Train-Val): {best_score:.4f}")
    print(f"{'='*60}")
    
    # Print saved snapshots
    print(f"\nSaved snapshots (top {num_snapshots}):")
    for score, ep, path in best_checkpoints:
        print(f"  Epoch {ep}: {score:.4f} -> {os.path.basename(path)}")
    
    if CONFIG["use_wandb"]:
        wandb.finish()
    
    # Run full validation using solution.py
    print(f"\n{'='*60}")
    print("Running full validation using solution.py...")
    print("(This matches the competition inference pipeline)")
    print(f"{'='*60}")
    
    solution_path = os.path.join(base_dir, "solution.py")
    if os.path.exists(solution_path):
        try:
            result = subprocess.run(
                ["python", solution_path],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.TimeoutExpired:
            print("Full validation timed out after 10 minutes")
        except Exception as e:
            print(f"Error running full validation: {e}")
    else:
        print(f"solution.py not found at {solution_path}")


if __name__ == "__main__":
    main()
