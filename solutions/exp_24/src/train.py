"""
exp_24: Training script for LiT-inspired LOB Transformer

Architecture: Patch Embedding → Causal Transformer → GRU → Regression Head
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

from model import LiTModel

# ============================================
# Configuration
# ============================================
CONFIG = {
    "experiment": "exp_24_lit_transformer",
    "input_dim": 32,        # Raw LOB features
    "embed_dim": 64,        # Transformer embedding dimension
    "patch_size": 1,        # Each timestep is a patch (for single-step inference)
    "num_transformer_layers": 2,
    "num_heads": 4,
    "ff_dim": 256,
    "gru_hidden": 128,
    "gru_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_steps": 99,     # First 99 steps not scored
    "use_pearson_loss": False,
    "use_wandb": True,
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
class WeightedPearsonLoss(nn.Module):
    """
    Weighted Pearson Correlation Loss matching competition metric.
    Weight: 2/3 * t0_corr + 1/3 * t1_corr
    """
    
    def __init__(self, t0_weight=2/3, t1_weight=1/3):
        super().__init__()
        self.t0_weight = t0_weight
        self.t1_weight = t1_weight
    
    def pearson_corr(self, pred, target):
        """Compute Pearson correlation."""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        cov = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        return cov / (pred_std * target_std)
    
    def forward(self, pred, target):
        # pred, target: (batch * seq, 2)
        t0_corr = self.pearson_corr(pred[:, 0], target[:, 0])
        t1_corr = self.pearson_corr(pred[:, 1], target[:, 1])
        
        weighted_corr = self.t0_weight * t0_corr + self.t1_weight * t1_corr
        return 1 - weighted_corr  # Minimize negative correlation


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
    """LOB Dataset with raw features."""
    
    def __init__(self, parquet_path):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        
        # Get sequence IDs
        self.seq_ids = self.df['seq_ix'].unique()
        print(f"Found {len(self.seq_ids)} sequences")
        
        # Feature and target columns (32 raw features)
        self.feature_cols = [c for c in self.df.columns 
                            if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        print(f"Features: {len(self.feature_cols)}, Targets: {len(self.target_cols)}")
        
        # Cache sequences
        print("Caching sequences...")
        self.data_cache = {}
        
        grouped = self.df.groupby('seq_ix')
        for seq_ix in tqdm(self.seq_ids, desc="Extracting sequences"):
            group = grouped.get_group(seq_ix)
            
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            need_pred = group['need_prediction'].values.astype(bool)
            
            # Normalize features (per-sequence z-score for prices/volumes)
            features = self._normalize_features(features)
            
            self.data_cache[seq_ix] = {
                'features': features,
                'targets': targets,
                'need_pred': need_pred
            }
        
        print(f"Cached {len(self.data_cache)} sequences")
    
    def _normalize_features(self, features):
        """Simple normalization: z-score per feature across sequence."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    
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
    
    features_padded = []
    targets_padded = []
    need_pred_padded = []
    lengths = []
    
    for item in batch:
        seq_len = item['features'].shape[0]
        lengths.append(seq_len)
        
        # Pad to max_len (pad at end)
        feat_pad = torch.zeros(max_len, item['features'].shape[1])
        feat_pad[:seq_len] = item['features']
        features_padded.append(feat_pad)
        
        tgt_pad = torch.zeros(max_len, 2)
        tgt_pad[:seq_len] = item['targets']
        targets_padded.append(tgt_pad)
        
        np_pad = torch.zeros(max_len, dtype=torch.bool)
        np_pad[:seq_len] = item['need_pred']
        need_pred_padded.append(np_pad)
    
    return {
        'features': torch.stack(features_padded),
        'targets': torch.stack(targets_padded),
        'need_pred': torch.stack(need_pred_padded),
        'lengths': torch.tensor(lengths)
    }


# ============================================
# Training Functions
# ============================================
def train_epoch(model, dataloader, optimizer, criterion, device, warmup_steps=99):
    model.train()
    total_loss = 0
    num_batches = 0
    
    all_preds_t0, all_preds_t1 = [], []
    all_targets_t0, all_targets_t1 = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        need_pred = batch['need_pred'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(features)
        
        # Mask: only score steps after warmup where need_prediction=True
        # Create step indices
        batch_size, seq_len, _ = predictions.shape
        step_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Valid mask: after warmup AND need_prediction
        valid_mask = (step_indices >= warmup_steps) & need_pred
        
        if valid_mask.sum() == 0:
            continue
        
        # Flatten and mask
        pred_flat = predictions.view(-1, 2)[valid_mask.view(-1)]
        target_flat = targets.view(-1, 2)[valid_mask.view(-1)]
        
        # Compute loss
        loss = criterion(pred_flat, target_flat)
        
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Collect for correlation calculation
        all_preds_t0.append(pred_flat[:, 0].detach().cpu())
        all_preds_t1.append(pred_flat[:, 1].detach().cpu())
        all_targets_t0.append(target_flat[:, 0].detach().cpu())
        all_targets_t1.append(target_flat[:, 1].detach().cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Calculate correlations
    if all_preds_t0:
        preds_t0 = torch.cat(all_preds_t0).numpy()
        preds_t1 = torch.cat(all_preds_t1).numpy()
        targets_t0 = torch.cat(all_targets_t0).numpy()
        targets_t1 = torch.cat(all_targets_t1).numpy()
        
        corr_t0 = np.corrcoef(preds_t0, targets_t0)[0, 1]
        corr_t1 = np.corrcoef(preds_t1, targets_t1)[0, 1]
    else:
        corr_t0, corr_t1 = 0.0, 0.0
    
    return avg_loss, corr_t0, corr_t1


def validate(model, dataloader, criterion, device, warmup_steps=99):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds_t0, all_preds_t1 = [], []
    all_targets_t0, all_targets_t1 = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            need_pred = batch['need_pred'].to(device)
            
            predictions, _ = model(features)
            
            batch_size, seq_len, _ = predictions.shape
            step_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            valid_mask = (step_indices >= warmup_steps) & need_pred
            
            if valid_mask.sum() == 0:
                continue
            
            pred_flat = predictions.view(-1, 2)[valid_mask.view(-1)]
            target_flat = targets.view(-1, 2)[valid_mask.view(-1)]
            
            loss = criterion(pred_flat, target_flat)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            all_preds_t0.append(pred_flat[:, 0].cpu())
            all_preds_t1.append(pred_flat[:, 1].cpu())
            all_targets_t0.append(target_flat[:, 0].cpu())
            all_targets_t1.append(target_flat[:, 1].cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Calculate correlations
    preds_t0 = torch.cat(all_preds_t0).numpy()
    preds_t1 = torch.cat(all_preds_t1).numpy()
    targets_t0 = torch.cat(all_targets_t0).numpy()
    targets_t1 = torch.cat(all_targets_t1).numpy()
    
    corr_t0 = np.corrcoef(preds_t0, targets_t0)[0, 1]
    corr_t1 = np.corrcoef(preds_t1, targets_t1)[0, 1]
    
    # Weighted score (competition metric)
    weighted_score = (2/3) * corr_t0 + (1/3) * corr_t1
    
    return avg_loss, corr_t0, corr_t1, weighted_score


# ============================================
# Main Training Loop
# ============================================
def main():
    # Initialize wandb
    if CONFIG["use_wandb"]:
        wandb.init(project=CONFIG["wandb_project"], name=CONFIG["experiment"], config=CONFIG)
    
    # Paths
    train_path = os.path.join(PROJECT_ROOT, "datasets", "train.parquet")
    valid_path = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    save_dir = os.path.dirname(SCRIPT_DIR)
    
    # Load datasets
    train_dataset = LOBDataset(train_path)
    valid_dataset = LOBDataset(valid_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    model = LiTModel(
        input_dim=CONFIG["input_dim"],
        embed_dim=CONFIG["embed_dim"],
        patch_size=CONFIG["patch_size"],
        num_transformer_layers=CONFIG["num_transformer_layers"],
        num_heads=CONFIG["num_heads"],
        ff_dim=CONFIG["ff_dim"],
        gru_hidden=CONFIG["gru_hidden"],
        gru_layers=CONFIG["gru_layers"],
        dropout=CONFIG["dropout"]
    ).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Loss and optimizer
    if CONFIG["use_pearson_loss"]:
        criterion = WeightedPearsonLoss()
    else:
        criterion = WeightedMSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )
    
    # Training loop
    best_val_score = -float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_t0, train_t1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, CONFIG["warmup_steps"]
        )
        train_score = (2/3) * train_t0 + (1/3) * train_t1
        
        # Validate
        val_loss, val_t0, val_t1, val_score = validate(
            model, valid_loader, criterion, DEVICE, CONFIG["warmup_steps"]
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, t0: {train_t0:.4f}, t1: {train_t1:.4f}, Score: {train_score:.4f}")
        print(f"Val Loss: {val_loss:.4f}, t0: {val_t0:.4f}, t1: {val_t1:.4f}, Score: {val_score:.4f}")
        
        # Log to wandb
        if CONFIG["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_t0_corr": train_t0,
                "train_t1_corr": train_t1,
                "train_score": train_score,
                "val_loss": val_loss,
                "val_t0_corr": val_t0,
                "val_t1_corr": val_t1,
                "val_score": val_score,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"*** New best model saved! Val Score: {val_score:.4f} ***")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Score: {best_val_score:.4f}")
    print(f"{'='*60}")
    
    if CONFIG["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
