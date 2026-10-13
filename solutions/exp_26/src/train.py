"""
exp_26: Training script for LOB Vision Transformer (LOB-ViT) - Causal Normalization

Improvements over exp_25b:
- Window size: 100 timesteps
- Causal normalization matching inference
- Attention pooling
- Mixed precision training (fp16) for memory efficiency

Architecture: Reshape → 2D Patch Embed → Transformer → Pool → GRU → Regression
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import heapq

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from model import LOBViTModel
from config import CONFIG, MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs

# ============================================
# Device Setup
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")

# Mixed precision setup - use bfloat16 on MPS, float16 on CUDA
USE_AMP = DEVICE.type in ["cuda", "mps"]
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "mps" else torch.float16
print(f"Mixed precision: {USE_AMP}, dtype: {AMP_DTYPE}")


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
    
    def __init__(self, mse_weight=0.5, pearson_weight=0.5):
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
# Dataset
# ============================================
class LOBDataset(Dataset):
    """LOB Dataset with causal normalization matching inference."""
    
    def __init__(self, parquet_path, norm_alpha=0.01):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.norm_alpha = norm_alpha
        
        # Get sequence IDs
        self.seq_ids = self.df['seq_ix'].unique()
        print(f"Found {len(self.seq_ids)} sequences")
        
        # Feature and target columns
        self.feature_cols = [c for c in self.df.columns 
                            if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        print(f"Features: {len(self.feature_cols)}, Targets: {len(self.target_cols)}")
        
        # Cache sequences
        print("Caching sequences with causal normalization...")
        self.data_cache = {}
        
        grouped = self.df.groupby('seq_ix')
        for seq_ix in tqdm(self.seq_ids, desc="Extracting sequences"):
            group = grouped.get_group(seq_ix)
            
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            need_pred = group['need_prediction'].values.astype(bool)
            
            # Causal normalization - matches inference exactly
            features = self._normalize_features_causal(features)
            
            self.data_cache[seq_ix] = {
                'features': features,
                'targets': targets,
                'need_pred': need_pred
            }
        
        print(f"Cached {len(self.data_cache)} sequences")
    
    def _normalize_features_causal(self, features):
        """
        Causal (online) normalization using EMA.
        
        This matches inference behavior exactly - only uses past information
        at each timestep. Prevents normalization leakage.
        
        Args:
            features: (seq_len, num_features) array
        Returns:
            normalized: (seq_len, num_features) array
        """
        seq_len, num_features = features.shape
        normalized = np.zeros_like(features)
        
        # Initialize running statistics with first observation
        running_mean = features[0].copy()
        running_var = np.ones(num_features, dtype=np.float32)  # Start with unit variance
        
        alpha = self.norm_alpha
        
        for t in range(seq_len):
            x = features[t]
            
            # Normalize using current running statistics
            std = np.sqrt(running_var + 1e-8)
            normalized[t] = (x - running_mean) / std
            
            # Update running statistics for next timestep (EMA)
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
    
    features_padded = []
    targets_padded = []
    need_pred_padded = []
    lengths = []
    
    for item in batch:
        seq_len = item['features'].shape[0]
        lengths.append(seq_len)
        
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
# Evaluation
# ============================================
def weighted_pearson_correlation(pred_t0, true_t0, pred_t1, true_t1):
    """Calculate weighted Pearson correlation (competition metric)."""
    def pearson(pred, true):
        pred = np.array(pred)
        true = np.array(true)
        pred_centered = pred - pred.mean()
        true_centered = true - true.mean()
        cov = (pred_centered * true_centered).sum()
        std_pred = np.sqrt((pred_centered ** 2).sum())
        std_true = np.sqrt((true_centered ** 2).sum())
        if std_pred < 1e-8 or std_true < 1e-8:
            return 0.0
        return cov / (std_pred * std_true)
    
    t0_corr = pearson(pred_t0, true_t0)
    t1_corr = pearson(pred_t1, true_t1)
    
    return {
        't0_corr': t0_corr,
        't1_corr': t1_corr,
        'weighted_score': (2/3) * t0_corr + (1/3) * t1_corr
    }


def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    all_pred_t0, all_true_t0 = [], []
    all_pred_t1, all_true_t1 = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            need_pred = batch['need_pred'].to(device)
            lengths = batch['lengths']
            
            # Forward pass
            predictions, _ = model.forward_full_sequence(features)
            
            # Extract predictions where need_prediction is True
            for i in range(features.shape[0]):
                seq_len = lengths[i].item()
                mask = need_pred[i, :seq_len]
                
                pred = predictions[i, :seq_len][mask]
                true = targets[i, :seq_len][mask]
                
                all_pred_t0.extend(pred[:, 0].cpu().numpy())
                all_true_t0.extend(true[:, 0].cpu().numpy())
                all_pred_t1.extend(pred[:, 1].cpu().numpy())
                all_true_t1.extend(true[:, 1].cpu().numpy())
    
    return weighted_pearson_correlation(all_pred_t0, all_true_t0, all_pred_t1, all_true_t1)


# ============================================
# Training
# ============================================
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        need_pred = batch['need_pred'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision (works for both CUDA with scaler and MPS without)
        if USE_AMP:
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE):
                predictions, _ = model.forward_full_sequence(features)
                
                # Flatten and mask
                mask = need_pred.view(-1)
                pred_flat = predictions.view(-1, 2)[mask]
                target_flat = targets.view(-1, 2)[mask]
                
                # Loss
                loss = criterion(pred_flat, target_flat)
            
            if scaler is not None:
                # CUDA with GradScaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS without GradScaler
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            # Standard fp32 training
            predictions, _ = model.forward_full_sequence(features)
            
            # Flatten and mask
            mask = need_pred.view(-1)
            pred_flat = predictions.view(-1, 2)[mask]
            target_flat = targets.view(-1, 2)[mask]
            
            # Loss
            loss = criterion(pred_flat, target_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def main():
    print("=" * 60)
    print("exp_26: LOB Vision Transformer - Causal Normalization")
    print("=" * 60)
    
    # Initialize wandb
    if CONFIG["use_wandb"]:
        wandb.init(project=CONFIG["wandb_project"], name=CONFIG["experiment"], config=CONFIG)
    
    # Load datasets with causal normalization
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
    
    # Create model using shared config
    model = LOBViTModel(**get_model_kwargs(for_inference=False)).to(DEVICE)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(mse_weight=0.7, pearson_weight=0.3)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Mixed precision scaler (only for CUDA, MPS uses autocast without scaler)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" and USE_AMP else None
    
    # Learning rate scheduler - warm restarts for snapshot ensemble
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader), T_mult=1
    )
    
    # Snapshot ensemble: keep track of top N checkpoints
    # Use min-heap with negative scores (to track max scores)
    num_snapshots = CONFIG["num_snapshots"]
    best_checkpoints = []  # List of (score, epoch, state_dict)
    best_score = -float('inf')
    best_epoch = 0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Train with mixed precision
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler, scaler)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Evaluate in fp32
        val_results = evaluate_model(model, valid_loader, DEVICE)
        print(f"Val t0_corr: {val_results['t0_corr']:.4f}")
        print(f"Val t1_corr: {val_results['t1_corr']:.4f}")
        print(f"Val Weighted Score: {val_results['weighted_score']:.4f}")
        
        # Log to wandb
        if CONFIG["use_wandb"]:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_t0_corr': val_results['t0_corr'],
                'val_t1_corr': val_results['t1_corr'],
                'val_weighted_score': val_results['weighted_score'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        current_score = val_results['weighted_score']
        
        # Track best overall
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
        
        # Snapshot ensemble: keep top N checkpoints
        state_dict_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if len(best_checkpoints) < num_snapshots:
            heapq.heappush(best_checkpoints, (current_score, epoch, state_dict_copy))
            print(f"*** Saved snapshot {len(best_checkpoints)}/{num_snapshots} (Score: {current_score:.4f}) ***")
        elif current_score > best_checkpoints[0][0]:
            # Replace worst checkpoint if current is better
            removed = heapq.heapreplace(best_checkpoints, (current_score, epoch, state_dict_copy))
            print(f"*** Replaced snapshot (removed epoch {removed[1]}, score {removed[0]:.4f}) ***")
            print(f"*** New snapshot from epoch {epoch} (Score: {current_score:.4f}) ***")
    
    # Save all snapshot checkpoints with epoch numbers
    output_dir = os.path.dirname(SCRIPT_DIR)
    
    # Sort by score (best first)
    best_checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Saving {len(best_checkpoints)} snapshot checkpoints:")
    for i, (score, ep, state_dict) in enumerate(best_checkpoints):
        # Save with epoch number for later weight averaging after full validation
        checkpoint_path = os.path.join(output_dir, f"best_model_ep_{ep}.pth")
        torch.save(state_dict, checkpoint_path)
        print(f"  Snapshot {i+1}: Epoch {ep}, Train-Val Score {score:.4f} -> {checkpoint_path}")
    
    # Also save the best as best_model.pth for compatibility
    if best_checkpoints:
        best_path = os.path.join(output_dir, "best_model.pth")
        torch.save(best_checkpoints[0][2], best_path)
        print(f"\nBest model (epoch {best_checkpoints[0][1]}) also saved to: {best_path}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Train-Val Score: {best_score:.4f} (Epoch {best_epoch})")
    print(f"Saved {len(best_checkpoints)} snapshots for ensemble")
    print(f"\nNOTE: Run full validation with ScorerStepByStep on each checkpoint,")
    print(f"then average weights of best performing checkpoints.")
    print(f"{'='*60}")
    
    if CONFIG["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
