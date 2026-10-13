"""
Training script for exp_12: Dual Stream T1-Focused Model

Specialized architecture with asymmetric heads for t0/t1 prediction.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS
from model import AsymmetricDualHeadGRU, count_parameters
from losses import HuberLossWeighted, compute_pearson_correlation
from features import LOBFeatureExtractor

# Optional: WandB for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")


def load_data(train_path, valid_path, seq_length=1000):
    """Load and prepare data."""
    print("Loading data...")
    
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Valid: {len(valid_df):,} samples")
    
    # Feature extraction
    extractor = LOBFeatureExtractor(seq_length=seq_length)
    
    X_train, y_train, m_train = extractor.fit_transform(train_df)
    X_valid, y_valid, m_valid = extractor.transform(valid_df)
    
    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Valid sequences: {X_valid.shape[0]}")
    
    return X_train, y_train, m_train, X_valid, y_valid, m_valid, extractor


def create_dataloaders(X_train, y_train, m_train, X_valid, y_valid, m_valid, batch_size):
    """Create PyTorch dataloaders."""
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(m_train).unsqueeze(-1)
    )
    
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid),
        torch.FloatTensor(y_valid),
        torch.FloatTensor(m_valid).unsqueeze(-1)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, valid_loader


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_loss_t0 = 0
    total_loss_t1 = 0
    n_batches = 0
    
    for batch_x, batch_y, batch_mask in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - no hidden state across shuffled batches
        # Each sequence in the batch is independent
        pred_t0, pred_t1, _ = model(batch_x, None)
        
        # Split targets
        target_t0 = batch_y[:, :, 0:1]
        target_t1 = batch_y[:, :, 1:2]
        
        # Compute loss
        loss, loss_t0, loss_t1 = criterion(
            pred_t0, pred_t1, 
            target_t0, target_t1, 
            batch_mask
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_t0 += loss_t0.item()
        total_loss_t1 += loss_t1.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'loss_t0': total_loss_t0 / n_batches,
        'loss_t1': total_loss_t1 / n_batches,
    }


@torch.no_grad()
def evaluate(model, valid_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    
    all_pred_t0 = []
    all_pred_t1 = []
    all_target_t0 = []
    all_target_t1 = []
    all_mask = []
    
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y, batch_mask in valid_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        # Forward pass - no hidden state across batches
        pred_t0, pred_t1, _ = model(batch_x, None)
        
        target_t0 = batch_y[:, :, 0:1]
        target_t1 = batch_y[:, :, 1:2]
        
        loss, _, _ = criterion(pred_t0, pred_t1, target_t0, target_t1, batch_mask)
        total_loss += loss.item()
        n_batches += 1
        
        all_pred_t0.append(pred_t0.cpu())
        all_pred_t1.append(pred_t1.cpu())
        all_target_t0.append(target_t0.cpu())
        all_target_t1.append(target_t1.cpu())
        all_mask.append(batch_mask.cpu())
    
    # Concatenate
    all_pred_t0 = torch.cat(all_pred_t0, dim=0)
    all_pred_t1 = torch.cat(all_pred_t1, dim=0)
    all_target_t0 = torch.cat(all_target_t0, dim=0)
    all_target_t1 = torch.cat(all_target_t1, dim=0)
    all_mask = torch.cat(all_mask, dim=0).bool()
    
    # Compute Pearson correlations
    pearson_t0 = compute_pearson_correlation(all_pred_t0, all_target_t0, all_mask)
    pearson_t1 = compute_pearson_correlation(all_pred_t1, all_target_t1, all_mask)
    
    pearson_avg = 0.5 * pearson_t0 + 0.5 * pearson_t1
    
    return {
        'loss': total_loss / n_batches,
        'pearson_t0': pearson_t0,
        'pearson_t1': pearson_t1,
        'pearson_avg': pearson_avg,
    }


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=8, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train(args):
    """Main training function."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, m_train, X_valid, y_valid, m_valid, extractor = load_data(
        DATA_CONFIG['train_path'],
        DATA_CONFIG['valid_path'],
        DATA_CONFIG['sequence_length']
    )
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        X_train, y_train, m_train,
        X_valid, y_valid, m_valid,
        TRAINING_CONFIG['batch_size']
    )
    
    # Create model
    model = AsymmetricDualHeadGRU(
        input_dim=MODEL_CONFIG['input_dim'],
        gru_hidden=MODEL_CONFIG['gru_hidden'],
        gru_layers=MODEL_CONFIG['gru_layers'],
        gru_dropout=MODEL_CONFIG['gru_dropout'],
        t0_hidden=MODEL_CONFIG['t0_hidden'],
        t0_dropout=MODEL_CONFIG['t0_dropout'],
        t1_hidden1=MODEL_CONFIG['t1_hidden1'],
        t1_hidden2=MODEL_CONFIG['t1_hidden2'],
        t1_dropout=MODEL_CONFIG['t1_dropout'],
    )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function - weighted toward t1 (t1 is the focus)
    criterion = HuberLossWeighted(
        t0_weight=TRAINING_CONFIG['t0_weight'],
        t1_weight=TRAINING_CONFIG['t1_weight'],
        delta=TRAINING_CONFIG['huber_delta']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['patience'],
        min_delta=TRAINING_CONFIG['min_delta'],
        mode='max'
    )
    
    # WandB logging
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=PATHS['wandb_project'],
            name=PATHS['wandb_name'],
            config={
                'model': MODEL_CONFIG,
                'training': TRAINING_CONFIG,
            }
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_score = 0
    epochs = args.epochs if args.epochs else TRAINING_CONFIG['epochs']
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            TRAINING_CONFIG['gradient_clip']
        )
        
        # Evaluate
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"  Pearson t0: {valid_metrics['pearson_t0']:.4f}")
        print(f"  Pearson t1: {valid_metrics['pearson_t1']:.4f}")
        print(f"  Pearson avg: {valid_metrics['pearson_avg']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # WandB logging
        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/loss_t0': train_metrics['loss_t0'],
                'train/loss_t1': train_metrics['loss_t1'],
                'valid/loss': valid_metrics['loss'],
                'valid/pearson_t0': valid_metrics['pearson_t0'],
                'valid/pearson_t1': valid_metrics['pearson_t1'],
                'valid/pearson_avg': valid_metrics['pearson_avg'],
                'lr': current_lr,
            })
        
        # Early stopping check
        should_save = early_stopping(valid_metrics['pearson_avg'])
        
        if should_save:
            best_score = valid_metrics['pearson_avg']
            print(f"  ✓ New best! Saving model...")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': valid_metrics,
                'model_config': MODEL_CONFIG,
                'normalization': extractor.get_normalization_params(),
            }, PATHS['model_save'])
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation score: {best_score:.4f}")
    print("=" * 60)
    
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()
    
    return best_score


def main():
    parser = argparse.ArgumentParser(description='Train Dual Stream T1-Focused model for exp_12')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
