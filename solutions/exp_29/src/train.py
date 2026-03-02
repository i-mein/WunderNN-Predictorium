"""
exp_29: Training script for LOBViT 2D Transformer

Key features:
- 2D reshaping of LOB data: (B, 12, T, 3)
- ViT-style patch embedding
- Causal normalization (EMA-based)
- Composite loss function
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
sys.path.insert(0, PROJECT_ROOT)

from model import LOBViT, reshape_raw_to_2d
from config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from src.utils import weighted_pearson_correlation

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")


# ============================================
# Loss Functions
# ============================================
class CompositeLoss(nn.Module):
    """Composite loss combining multiple objectives."""
    
    def __init__(self, weights: dict, huber_delta: float = 0.3):
        super().__init__()
        self.weights = weights
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        losses = {}
        
        # Standard losses
        losses["mse"] = self.mse(pred, target)
        losses["huber"] = self.huber(pred, target)
        losses["l1"] = self.l1(pred, target)
        
        # Weighted MSE (emphasize larger movements)
        weights = torch.abs(target) + 1e-8
        weights = weights / weights.sum() * target.numel()
        losses["weighted_mse"] = (weights * (pred - target) ** 2).mean()
        
        # Temporal consistency (smooth predictions)
        if pred.size(0) > 1:
            pred_diff = torch.diff(pred, dim=0)
            target_diff = torch.diff(target, dim=0)
            losses["temporal"] = self.mse(pred_diff, target_diff)
        else:
            losses["temporal"] = torch.tensor(0.0, device=pred.device)
        
        # Pearson correlation loss (per-target)
        pearson_losses = []
        for i in range(pred.size(1)):
            p, t = pred[:, i], target[:, i]
            p_centered = p - p.mean()
            t_centered = t - t.mean()
            cov = (p_centered * t_centered).mean()
            p_std = p_centered.std() + 1e-8
            t_std = t_centered.std() + 1e-8
            corr = cov / (p_std * t_std)
            pearson_losses.append(1 - corr)
        losses["pearson"] = sum(pearson_losses) / len(pearson_losses)
        
        # Combine
        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        
        return total, {k: v.item() for k, v in losses.items()}


# ============================================
# Dataset
# ============================================
class LOB2DDataset(Dataset):
    """Dataset that provides 2D-reshaped LOB sequences."""
    
    def __init__(
        self,
        parquet_path: str,
        seq_ids: list = None,
        context_len: int = 100,
        augment: bool = False,
        noise_std: float = 0.005,
        norm_alpha: float = 0.01,
    ):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.context_len = context_len
        self.augment = augment
        self.noise_std = noise_std
        self.norm_alpha = norm_alpha
        
        if seq_ids is not None:
            self.seq_ids = seq_ids
        else:
            self.seq_ids = self.df['seq_ix'].unique().tolist()
        
        # Feature columns (32 raw)
        self.feature_cols = [
            c for c in self.df.columns
            if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
        ]
        self.target_cols = ['t0', 't1']
        
        # Cache sequences
        print("Caching sequences with causal normalization...")
        self.data_cache = {}
        self._preprocess_sequences()
    
    def _causal_normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply causal EMA normalization."""
        normalized = np.zeros_like(features)
        running_mean = None
        running_var = None
        alpha = self.norm_alpha
        
        for t in range(features.shape[0]):
            state = features[t]
            
            if running_mean is None:
                running_mean = state.copy()
                running_var = np.ones_like(state, dtype=np.float32)
                normalized[t] = np.zeros_like(state)
            else:
                std = np.sqrt(running_var + 1e-8)
                normalized[t] = (state - running_mean) / std
            
            delta = state - running_mean
            running_mean = running_mean + alpha * delta
            running_var = (1 - alpha) * running_var + alpha * (delta ** 2)
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _preprocess_sequences(self):
        grouped = self.df.groupby('seq_ix')
        
        for seq_ix in tqdm(self.seq_ids, desc="Processing sequences"):
            group = grouped.get_group(seq_ix)
            raw_features = group[self.feature_cols].values.astype(np.float32)
            normalized = self._causal_normalize(raw_features)
            targets = group[self.target_cols].values.astype(np.float32)
            need_pred = group['need_prediction'].values
            
            self.data_cache[seq_ix] = (normalized, targets, need_pred)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        normalized, targets, need_pred = self.data_cache[seq_ix]
        
        contexts_list = []
        targets_list = []
        
        for t in range(len(normalized)):
            if not need_pred[t]:
                continue
            
            start_idx = max(0, t - self.context_len + 1)
            context = normalized[start_idx:t+1]
            
            # Pad if needed
            if context.shape[0] < self.context_len:
                pad_len = self.context_len - context.shape[0]
                context = np.pad(context, ((pad_len, 0), (0, 0)), mode='edge')
            
            # Augmentation
            if self.augment and np.random.rand() < 0.5:
                noise = np.random.randn(*context.shape) * self.noise_std
                context = context + noise.astype(np.float32)
            
            contexts_list.append(context)
            targets_list.append(targets[t])
        
        if len(contexts_list) == 0:
            return (
                torch.zeros(1, self.context_len, 32),
                torch.zeros(1, 2),
            )
        
        contexts = np.stack(contexts_list, axis=0)
        targets_arr = np.stack(targets_list, axis=0)
        
        return (
            torch.tensor(contexts, dtype=torch.float32),
            torch.tensor(targets_arr, dtype=torch.float32),
        )


def collate_fn(batch):
    contexts = torch.cat([b[0] for b in batch], dim=0)
    targets = torch.cat([b[1] for b in batch], dim=0)
    return contexts, targets


# ============================================
# Training
# ============================================
def train():
    """Main training function."""
    
    if TRAIN_CONFIG["use_wandb"]:
        wandb.init(
            project=TRAIN_CONFIG["wandb_project"],
            name=TRAIN_CONFIG["experiment"],
            config={**MODEL_CONFIG, **TRAIN_CONFIG},
        )
    
    # Paths
    train_file = os.path.join(PROJECT_ROOT, "datasets", "train.parquet")
    valid_file = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    save_dir = os.path.dirname(SCRIPT_DIR)
    
    # Load data
    print("Loading data...")
    df_train = pd.read_parquet(train_file)
    train_seqs = df_train['seq_ix'].unique().tolist()
    
    df_valid = pd.read_parquet(valid_file)
    val_seqs = df_valid['seq_ix'].unique().tolist()
    
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}")
    
    # Datasets
    context_len = MODEL_CONFIG["context_len"]
    norm_alpha = TRAIN_CONFIG["norm_ema_alpha"]
    
    train_dataset = LOB2DDataset(
        train_file,
        seq_ids=train_seqs,
        context_len=context_len,
        augment=True,
        noise_std=TRAIN_CONFIG["noise_std"],
        norm_alpha=norm_alpha,
    )
    
    val_dataset = LOB2DDataset(
        valid_file,
        seq_ids=val_seqs,
        context_len=context_len,
        augment=False,
        norm_alpha=norm_alpha,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"],
    )
    
    # Model
    print("Initializing LOBViT model...")
    model = LOBViT(**get_model_kwargs(for_inference=False))
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    
    # Scheduler
    total_steps = len(train_loader) * TRAIN_CONFIG["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_CONFIG["learning_rate"] * 4,
        total_steps=total_steps,
        pct_start=TRAIN_CONFIG["warmup_pct"],
        anneal_strategy="cos",
    )
    
    # Loss
    loss_fn = CompositeLoss(
        weights=TRAIN_CONFIG["loss_weights"],
        huber_delta=TRAIN_CONFIG["huber_delta"],
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_corr = -float('inf')
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        for raw_contexts, targets in pbar:
            # Reshape to 2D: (B, T, 32) → (B, 12, T, 3)
            contexts_2d = reshape_raw_to_2d(raw_contexts, context_len)
            contexts_2d = contexts_2d.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            predictions = model(contexts_2d)
            loss, loss_dict = loss_fn(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
        
        train_loss /= train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for raw_contexts, targets in val_loader:
                contexts_2d = reshape_raw_to_2d(raw_contexts, context_len)
                contexts_2d = contexts_2d.to(DEVICE)
                targets = targets.to(DEVICE)
                
                predictions = model(contexts_2d)
                loss, _ = loss_fn(predictions, targets)
                
                val_loss += loss.item()
                val_steps += 1
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        val_loss /= val_steps
        
        # Compute correlations
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        t0_corr = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
        t1_corr = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
        weighted_corr = (2/3) * t0_corr + (1/3) * t1_corr
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Val Corr: t0={t0_corr:.4f}, t1={t1_corr:.4f}, weighted={weighted_corr:.4f}")
        
        if TRAIN_CONFIG["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_t0_corr": t0_corr,
                "val_t1_corr": t1_corr,
                "val_weighted_corr": weighted_corr,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        
        # Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  → Saved best model (loss={val_loss:.4f})")
        else:
            patience_counter += 1
        
        if weighted_corr > best_val_corr:
            best_val_corr = weighted_corr
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_corr.pth"))
            print(f"  → Saved best model by correlation (corr={weighted_corr:.4f})")
        
        if patience_counter >= TRAIN_CONFIG["patience"]:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val correlation: {best_val_corr:.4f}")
    
    if TRAIN_CONFIG["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    train()
