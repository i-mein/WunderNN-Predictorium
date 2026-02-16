"""
Training script for exp_11b: Dual Causal Transformers

Two separate transformer models with different optimizers and hyperparameters.
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

from config import MODEL_T0_CONFIG, MODEL_T1_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS
from model import DualTransformerModel, CausalTransformerSingle, count_parameters
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
        pin_memory=False  # MPS doesn't support pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, valid_loader


def train_epoch(model_t0, model_t1, train_loader, criterion, optimizer_t0, optimizer_t1, device, gradient_clip):
    """Train for one epoch with separate optimizers."""
    model_t0.train()
    model_t1.train()
    
    total_loss = 0
    total_loss_t0 = 0
    total_loss_t1 = 0
    n_batches = 0
    
    for batch_x, batch_y, batch_mask in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        # Zero gradients
        optimizer_t0.zero_grad()
        optimizer_t1.zero_grad()
        
        # Forward pass through both models
        pred_t0 = model_t0(batch_x)
        pred_t1 = model_t1(batch_x)
        
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
            torch.nn.utils.clip_grad_norm_(model_t0.parameters(), gradient_clip)
            torch.nn.utils.clip_grad_norm_(model_t1.parameters(), gradient_clip)
        
        # Update both optimizers
        optimizer_t0.step()
        optimizer_t1.step()
        
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
def evaluate(model_t0, model_t1, valid_loader, criterion, device):
    """Evaluate on validation set."""
    model_t0.eval()
    model_t1.eval()
    
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
        
        # Forward pass
        pred_t0 = model_t0(batch_x)
        pred_t1 = model_t1(batch_x)
        
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
    
    # Create two separate models
    model_t0 = CausalTransformerSingle(**MODEL_T0_CONFIG).to(device)
    model_t1 = CausalTransformerSingle(**MODEL_T1_CONFIG).to(device)
    
    print(f"Model T0 parameters: {count_parameters(model_t0):,}")
    print(f"Model T1 parameters: {count_parameters(model_t1):,}")
    print(f"Total parameters: {count_parameters(model_t0) + count_parameters(model_t1):,}")
    
    # Loss function
    criterion = HuberLossWeighted(
        t0_weight=TRAINING_CONFIG['t0_weight'],
        t1_weight=TRAINING_CONFIG['t1_weight'],
        delta=TRAINING_CONFIG['huber_delta']
    )
    
    # Separate optimizers with different learning rates
    optimizer_t0 = torch.optim.AdamW(
        model_t0.parameters(),
        lr=TRAINING_CONFIG['lr_t0'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    optimizer_t1 = torch.optim.AdamW(
        model_t1.parameters(),
        lr=TRAINING_CONFIG['lr_t1'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Separate schedulers
    scheduler_t0 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_t0, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    scheduler_t1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_t1, T_0=15, T_mult=2, eta_min=1e-6
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
                'model_t0': MODEL_T0_CONFIG,
                'model_t1': MODEL_T1_CONFIG,
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
            model_t0, model_t1, train_loader, criterion, 
            optimizer_t0, optimizer_t1, device,
            TRAINING_CONFIG['gradient_clip']
        )
        
        # Evaluate
        valid_metrics = evaluate(model_t0, model_t1, valid_loader, criterion, device)
        
        # Update schedulers
        scheduler_t0.step()
        scheduler_t1.step()
        
        lr_t0 = optimizer_t0.param_groups[0]['lr']
        lr_t1 = optimizer_t1.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (t0: {train_metrics['loss_t0']:.4f}, t1: {train_metrics['loss_t1']:.4f})")
        print(f"  Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"  Pearson t0: {valid_metrics['pearson_t0']:.4f}")
        print(f"  Pearson t1: {valid_metrics['pearson_t1']:.4f}")
        print(f"  Pearson avg: {valid_metrics['pearson_avg']:.4f}")
        print(f"  LR t0: {lr_t0:.6f}, LR t1: {lr_t1:.6f}")
        
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
                'lr_t0': lr_t0,
                'lr_t1': lr_t1,
            })
        
        # Early stopping check
        should_save = early_stopping(valid_metrics['pearson_avg'])
        
        if should_save:
            best_score = valid_metrics['pearson_avg']
            print(f"  ✓ New best! Saving models...")
            
            # Save T0 model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_t0.state_dict(),
                'optimizer_state_dict': optimizer_t0.state_dict(),
                'metrics': valid_metrics,
                'model_config': MODEL_T0_CONFIG,
                'normalization': extractor.get_normalization_params(),
            }, PATHS['model_t0_save'])
            
            # Save T1 model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_t1.state_dict(),
                'optimizer_state_dict': optimizer_t1.state_dict(),
                'metrics': valid_metrics,
                'model_config': MODEL_T1_CONFIG,
                'normalization': extractor.get_normalization_params(),
            }, PATHS['model_t1_save'])
        
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
    parser = argparse.ArgumentParser(description='Train Dual Causal Transformers for exp_11b')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
