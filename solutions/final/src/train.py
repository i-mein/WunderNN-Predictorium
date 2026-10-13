"""
exp_21: Multi-Stream GRU with Temporal Attention
Training module
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.parquet as pq
from tqdm import tqdm
import wandb
from pathlib import Path

from model import MultiStreamAttentionModel


def load_data(train_path: str, valid_path: str, seq_len: int = 100):
    """Load and prepare training and validation data."""
    
    print("Loading training data...")
    train_df = pq.read_table(train_path).to_pandas()
    
    print("Loading validation data...")
    valid_df = pq.read_table(valid_path).to_pandas()
    
    # Feature columns
    feature_cols = [f'p{i}' for i in range(12)] + \
                   [f'v{i}' for i in range(12)] + \
                   [f'dp{i}' for i in range(4)] + \
                   [f'dv{i}' for i in range(4)]
    
    # Get sequences using seq_ix
    def extract_sequences(df):
        sequences = []
        targets_t0 = []
        targets_t1 = []
        
        seq_ids = df['seq_ix'].unique()
        for seq_ix in tqdm(seq_ids, desc="Extracting sequences"):
            group = df[df['seq_ix'] == seq_ix]
            if len(group) >= seq_len:
                features = group[feature_cols].values[:seq_len].astype(np.float32)
                t0 = group['t0'].values[:seq_len].astype(np.float32)
                t1 = group['t1'].values[:seq_len].astype(np.float32)
                
                sequences.append(features)
                targets_t0.append(t0)
                targets_t1.append(t1)
        
        return np.array(sequences), np.array(targets_t0), np.array(targets_t1)
    
    print("Extracting training sequences...")
    X_train, y_train_t0, y_train_t1 = extract_sequences(train_df)
    
    print("Extracting validation sequences...")
    X_valid, y_valid_t0, y_valid_t1 = extract_sequences(valid_df)
    
    print(f"Training: {len(X_train)} sequences")
    print(f"Validation: {len(X_valid)} sequences")
    
    return (
        torch.tensor(X_train),
        torch.tensor(y_train_t0),
        torch.tensor(y_train_t1),
        torch.tensor(X_valid),
        torch.tensor(y_valid_t0),
        torch.tensor(y_valid_t1),
    )


class WeightedMSELoss(nn.Module):
    """Weighted MSE with higher weight on extreme values."""
    
    def __init__(self, extreme_weight=2.0, threshold=0.5):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.threshold = threshold
    
    def forward(self, pred, target):
        diff = pred - target
        weights = torch.where(
            target.abs() > self.threshold,
            torch.tensor(self.extreme_weight, device=pred.device),
            torch.tensor(1.0, device=pred.device),
        )
        return (weights * diff ** 2).mean()


def pearson_correlation(pred, target):
    """Compute Pearson correlation."""
    pred = pred.flatten()
    target = target.flatten()
    
    # Remove NaN
    mask = ~(torch.isnan(pred) | torch.isnan(target))
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) < 2:
        return torch.tensor(0.0)
    
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    numerator = (pred_centered * target_centered).sum()
    denominator = (pred_centered.pow(2).sum() * target_centered.pow(2).sum()).sqrt()
    
    return numerator / (denominator + 1e-8)


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_corr_t0 = 0
    total_corr_t1 = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for X, y_t0, y_t1 in pbar:
        X = X.to(device)
        y_t0 = y_t0.to(device)
        y_t1 = y_t1.to(device)
        
        optimizer.zero_grad()
        
        pred_t0, pred_t1, _ = model(X)
        
        loss_t0 = criterion(pred_t0, y_t0)
        loss_t1 = criterion(pred_t1, y_t1)
        loss = loss_t0 + loss_t1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            total_corr_t0 += pearson_correlation(pred_t0, y_t0).item()
            total_corr_t1 += pearson_correlation(pred_t1, y_t1).item()
        
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'corr_t0': f'{total_corr_t0/num_batches:.4f}',
            'corr_t1': f'{total_corr_t1/num_batches:.4f}',
        })
    
    return {
        'loss': total_loss / num_batches,
        'corr_t0': total_corr_t0 / num_batches,
        'corr_t1': total_corr_t1 / num_batches,
    }


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_pred_t0 = []
    all_pred_t1 = []
    all_target_t0 = []
    all_target_t1 = []
    
    with torch.no_grad():
        for X, y_t0, y_t1 in tqdm(dataloader, desc="Validating"):
            X = X.to(device)
            y_t0 = y_t0.to(device)
            y_t1 = y_t1.to(device)
            
            pred_t0, pred_t1, _ = model(X)
            
            loss_t0 = criterion(pred_t0, y_t0)
            loss_t1 = criterion(pred_t1, y_t1)
            loss = loss_t0 + loss_t1
            
            total_loss += loss.item()
            
            all_pred_t0.append(pred_t0.cpu())
            all_pred_t1.append(pred_t1.cpu())
            all_target_t0.append(y_t0.cpu())
            all_target_t1.append(y_t1.cpu())
    
    pred_t0 = torch.cat(all_pred_t0)
    pred_t1 = torch.cat(all_pred_t1)
    target_t0 = torch.cat(all_target_t0)
    target_t1 = torch.cat(all_target_t1)
    
    corr_t0 = pearson_correlation(pred_t0, target_t0).item()
    corr_t1 = pearson_correlation(pred_t1, target_t1).item()
    
    return {
        'loss': total_loss / len(dataloader),
        'corr_t0': corr_t0,
        'corr_t1': corr_t1,
        'corr_avg': (corr_t0 + corr_t1) / 2,
    }


def main():
    # Hyperparameters
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 15,
        'price_hidden': 48,
        'price_output': 64,
        'volume_hidden': 48,
        'volume_output': 64,
        'delta_hidden': 32,
        'delta_output': 32,
        'eng_hidden': 32,
        'eng_output': 48,
        'head_hidden': 128,
        'dropout': 0.2,
        'num_attn_heads': 4,
        'extreme_weight': 2.0,
        'extreme_threshold': 0.5,
    }
    
    # Initialize wandb
    wandb.init(project="wundernn", name="exp_21_multistream_attention", config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train_t0, y_train_t1, X_valid, y_valid_t0, y_valid_t1 = load_data(
        'datasets/train.parquet',
        'datasets/valid.parquet',
    )
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train_t0, y_train_t1)
    valid_dataset = TensorDataset(X_valid, y_valid_t0, y_valid_t1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    model = MultiStreamAttentionModel(
        price_hidden=config['price_hidden'],
        price_output=config['price_output'],
        volume_hidden=config['volume_hidden'],
        volume_output=config['volume_output'],
        delta_hidden=config['delta_hidden'],
        delta_output=config['delta_output'],
        eng_hidden=config['eng_hidden'],
        eng_output=config['eng_output'],
        head_hidden=config['head_hidden'],
        dropout=config['dropout'],
        num_attn_heads=config['num_attn_heads'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = WeightedMSELoss(
        extreme_weight=config['extreme_weight'],
        threshold=config['extreme_threshold'],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    # Training loop
    best_corr = -float('inf')
    save_path = Path(__file__).parent.parent / 'best_model.pth'
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 60)
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        valid_metrics = validate(model, valid_loader, criterion, device)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Corr T0: {train_metrics['corr_t0']:.4f}, "
              f"Corr T1: {train_metrics['corr_t1']:.4f}")
        print(f"Valid - Loss: {valid_metrics['loss']:.4f}, "
              f"Corr T0: {valid_metrics['corr_t0']:.4f}, "
              f"Corr T1: {valid_metrics['corr_t1']:.4f}, "
              f"Avg: {valid_metrics['corr_avg']:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/corr_t0': train_metrics['corr_t0'],
            'train/corr_t1': train_metrics['corr_t1'],
            'valid/loss': valid_metrics['loss'],
            'valid/corr_t0': valid_metrics['corr_t0'],
            'valid/corr_t1': valid_metrics['corr_t1'],
            'valid/corr_avg': valid_metrics['corr_avg'],
        })
        
        # Save best model
        if valid_metrics['corr_avg'] > best_corr:
            best_corr = valid_metrics['corr_avg']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_corr_avg': valid_metrics['corr_avg'],
                'valid_corr_t0': valid_metrics['corr_t0'],
                'valid_corr_t1': valid_metrics['corr_t1'],
                'config': config,
            }, save_path)
            print(f"  -> New best model saved! Corr avg: {best_corr:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Training complete. Best validation correlation: {best_corr:.4f}")
    print(f"Model saved to: {save_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
