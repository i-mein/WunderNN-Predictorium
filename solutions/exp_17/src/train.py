"""
Exp 17: Training script for Enhanced Hybrid Model

Improvements over exp_13:
1. Lower learning rate with longer training
2. Gradient accumulation for larger effective batch size
3. Cosine annealing with warm restarts
4. Huber loss for robustness to outliers
5. Higher t1 weight (2.0 vs 1.5)
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from solutions.exp_17.src.model import EnhancedHybridModel


class Config:
    # Data
    SEQ_LEN = 1000
    WARMUP_STEPS = 99
    
    # Model
    INPUT_DIM = 32
    D_MODEL = 160
    N_HEADS = 8
    N_TRANSFORMER_LAYERS = 3
    GRU_HIDDEN = 192
    GRU_LAYERS = 2
    DROPOUT = 0.15
    
    # Training
    BATCH_SIZE = 24  # Smaller batch, use gradient accumulation
    ACCUMULATION_STEPS = 2  # Effective batch size = 48
    EPOCHS = 40
    LEARNING_RATE = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    
    # Loss weighting
    T0_WEIGHT = 1.0
    T1_WEIGHT = 2.0  # Even more weight on t1
    
    # Early stopping
    PATIENCE = 12
    
    DEVICE = None
    
    @classmethod
    def get_device(cls):
        if cls.DEVICE is None:
            if torch.backends.mps.is_available():
                cls.DEVICE = torch.device("mps")
            elif torch.cuda.is_available():
                cls.DEVICE = torch.device("cuda")
            else:
                cls.DEVICE = torch.device("cpu")
        return cls.DEVICE


class LOBDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.seq_ids = self.df['seq_ix'].unique()
        
        self.feature_cols = [f'p{i}' for i in range(12)] + \
                           [f'v{i}' for i in range(12)] + \
                           [f'dp{i}' for i in range(4)] + \
                           [f'dv{i}' for i in range(4)]
        self.target_cols = ['t0', 't1']
        
        print(f"Found {len(self.seq_ids)} sequences")
        
        self.data_cache = {}
        grouped = self.df.groupby('seq_ix')
        
        for seq_ix in tqdm(self.seq_ids, desc="Caching sequences"):
            group = grouped.get_group(seq_ix)
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            need_pred = group['need_prediction'].values.astype(bool)
            self.data_cache[seq_ix] = (features, targets, need_pred)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets, need_pred = self.data_cache[seq_ix]
        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(need_pred),
        )


def weighted_pearson_correlation(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return 0.0
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    num = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    den = np.sqrt(np.sum((y_true - mean_true)**2) * np.sum((y_pred - mean_pred)**2))
    
    if den < 1e-8:
        return 0.0
    
    return num / den


class CombinedLoss(nn.Module):
    """Combined loss with Huber for robustness."""
    
    def __init__(self, t0_weight=1.0, t1_weight=2.0):
        super().__init__()
        self.t0_weight = t0_weight
        self.t1_weight = t1_weight
        self.huber = nn.SmoothL1Loss(reduction='none', beta=1.0)
    
    def forward(self, pred_t0, pred_t1, target_t0, target_t1, mask):
        # Use Huber loss for robustness to outliers
        loss_t0 = self.huber(pred_t0, target_t0)
        loss_t1 = self.huber(pred_t1, target_t1)
        
        # Apply mask
        loss_t0 = (loss_t0 * mask.float()).sum() / (mask.sum() + 1e-8)
        loss_t1 = (loss_t1 * mask.float()).sum() / (mask.sum() + 1e-8)
        
        total_loss = self.t0_weight * loss_t0 + self.t1_weight * loss_t1
        
        return total_loss, loss_t0, loss_t1


def train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=1.0, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_loss_t0 = 0
    total_loss_t1 = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for i, (features, targets, need_pred) in enumerate(pbar):
        features = features.to(device)
        targets = targets.to(device)
        need_pred = need_pred.to(device)
        
        pred_t0, pred_t1, _ = model(features, hidden=None)
        
        target_t0 = targets[..., 0]
        target_t1 = targets[..., 1]
        
        loss, loss_t0, loss_t1 = criterion(pred_t0, pred_t1, target_t0, target_t1, need_pred)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_loss_t0 += loss_t0.item()
        total_loss_t1 += loss_t1.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            't0': f'{loss_t0.item():.4f}',
            't1': f'{loss_t1.item():.4f}'
        })
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    n = len(train_loader)
    return total_loss / n, total_loss_t0 / n, total_loss_t1 / n


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    
    total_loss = 0
    all_pred_t0, all_pred_t1 = [], []
    all_true_t0, all_true_t1 = [], []
    
    for features, targets, need_pred in tqdm(val_loader, desc="Validating"):
        features = features.to(device)
        targets = targets.to(device)
        need_pred = need_pred.to(device)
        
        pred_t0, pred_t1, _ = model(features, hidden=None)
        
        target_t0 = targets[..., 0]
        target_t1 = targets[..., 1]
        
        loss, _, _ = criterion(pred_t0, pred_t1, target_t0, target_t1, need_pred)
        total_loss += loss.item()
        
        mask = need_pred.cpu().numpy()
        pred_t0_np = pred_t0.cpu().numpy()
        pred_t1_np = pred_t1.cpu().numpy()
        target_t0_np = target_t0.cpu().numpy()
        target_t1_np = target_t1.cpu().numpy()
        
        for i in range(features.shape[0]):
            m = mask[i]
            all_pred_t0.extend(pred_t0_np[i, m])
            all_pred_t1.extend(pred_t1_np[i, m])
            all_true_t0.extend(target_t0_np[i, m])
            all_true_t1.extend(target_t1_np[i, m])
    
    val_loss = total_loss / len(val_loader)
    
    pearson_t0 = weighted_pearson_correlation(all_true_t0, all_pred_t0)
    pearson_t1 = weighted_pearson_correlation(all_true_t1, all_pred_t1)
    pearson_avg = (pearson_t0 + pearson_t1) / 2
    
    return val_loss, pearson_t0, pearson_t1, pearson_avg


def main():
    device = Config.get_device()
    print(f"Using device: {device}")
    
    wandb.init(
        project="WunderNN-Predictorium",
        name="exp_17_enhanced_hybrid",
        config={
            "model": "EnhancedHybridModel",
            "d_model": Config.D_MODEL,
            "n_heads": Config.N_HEADS,
            "n_transformer_layers": Config.N_TRANSFORMER_LAYERS,
            "gru_hidden": Config.GRU_HIDDEN,
            "gru_layers": Config.GRU_LAYERS,
            "dropout": Config.DROPOUT,
            "batch_size": Config.BATCH_SIZE,
            "accumulation_steps": Config.ACCUMULATION_STEPS,
            "effective_batch_size": Config.BATCH_SIZE * Config.ACCUMULATION_STEPS,
            "lr": Config.LEARNING_RATE,
            "epochs": Config.EPOCHS,
            "t0_weight": Config.T0_WEIGHT,
            "t1_weight": Config.T1_WEIGHT,
        }
    )
    
    print("Loading data...")
    train_dataset = LOBDataset("datasets/train.parquet")
    valid_dataset = LOBDataset("datasets/valid.parquet")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type != 'mps' else False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type != 'mps' else False,
    )
    
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Valid: {len(valid_dataset)} sequences")
    
    model = EnhancedHybridModel(
        input_dim=Config.INPUT_DIM,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_transformer_layers=Config.N_TRANSFORMER_LAYERS,
        gru_hidden=Config.GRU_HIDDEN,
        gru_layers=Config.GRU_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    wandb.config.update({"total_params": total_params})
    
    criterion = CombinedLoss(t0_weight=Config.T0_WEIGHT, t1_weight=Config.T1_WEIGHT)
    
    # Layer-wise learning rate decay
    no_decay = ['bias', 'LayerNorm.weight', 'norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)
    
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=Config.MIN_LR,
    )
    
    best_val_score = -float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    for epoch in range(Config.EPOCHS):
        train_loss, train_t0, train_t1 = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            Config.GRAD_CLIP, Config.ACCUMULATION_STEPS
        )
        
        val_loss, pearson_t0, pearson_t1, pearson_avg = validate(
            model, valid_loader, criterion, device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} (t0: {train_t0:.4f}, t1: {train_t1:.4f})")
        print(f"  Valid Loss: {val_loss:.4f}")
        print(f"  Pearson t0: {pearson_t0:.4f}")
        print(f"  Pearson t1: {pearson_t1:.4f}")
        print(f"  Pearson avg: {pearson_avg:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/loss_t0": train_t0,
            "train/loss_t1": train_t1,
            "valid/loss": val_loss,
            "valid/pearson_t0": pearson_t0,
            "valid/pearson_t1": pearson_t1,
            "valid/pearson_avg": pearson_avg,
            "lr": current_lr,
        })
        
        if pearson_avg > best_val_score:
            best_val_score = pearson_avg
            patience_counter = 0
            
            save_path = Path(__file__).parent.parent / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': best_val_score,
                'pearson_t0': pearson_t0,
                'pearson_t1': pearson_t1,
            }, save_path)
            print(f"  ✓ New best! Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best score: {best_val_score:.4f}")
    print("=" * 60)
    
    wandb.finish()


if __name__ == "__main__":
    main()
