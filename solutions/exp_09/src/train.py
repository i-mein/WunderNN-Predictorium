"""
Training script for exp_09: Dual-Head GRU with Pearson Loss.

Key differences from exp_06:
- Dual-head model (separate t0/t1 predictions)
- Pearson correlation loss (direct metric optimization)
- Same leakage-free features and validation
"""
import os, sys, gc
import math, random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config as CFG
from model import DualHeadGRU
from features import generate_features, get_feature_columns
from losses import combined_loss, pearson_correlation_loss

# Load environment variables
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Root directory (project root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# Utilities

def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_pearson_correlation(y_true, y_pred):
    """Calculate Pearson correlation."""
    from scipy.stats import pearsonr
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


class LOBDataset(Dataset):
    """Dataset for Limit Order Book sequences."""
    
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print("Generating features...")
        df = generate_features(df)
        
        self.feature_cols = get_feature_columns(df)
        print(f"Feature count: {len(self.feature_cols)}")
        
        self.target_cols = ['t0', 't1']
        self.seq_ids = df['seq_ix'].unique()
        
        # Cache sequences for faster loading
        self.data_cache = {}
        grouped = df.groupby('seq_ix')
        
        for seq_ix, group in tqdm(grouped, desc="Caching sequences"):
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            self.data_cache[seq_ix] = (features, targets)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets = self.data_cache[seq_ix]
        return torch.from_numpy(features), torch.from_numpy(targets)


def validate(model, val_loader, device):
    """
    CRITICAL: Leakage-free validation.
    Evaluates ALL timesteps (no warm-up period).
    """
    model.eval()
    all_preds_t0 = []
    all_preds_t1 = []
    all_targets_t0 = []
    all_targets_t1 = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred_t0, pred_t1, _ = model(X)
            
            # CRITICAL: Evaluate ALL timesteps
            all_preds_t0.append(pred_t0.cpu().numpy())
            all_preds_t1.append(pred_t1.cpu().numpy())
            all_targets_t0.append(y[:, :, 0:1].cpu().numpy())
            all_targets_t1.append(y[:, :, 1:2].cpu().numpy())
    
    # Flatten
    all_preds_t0 = np.concatenate(all_preds_t0, axis=0).flatten()
    all_preds_t1 = np.concatenate(all_preds_t1, axis=0).flatten()
    all_targets_t0 = np.concatenate(all_targets_t0, axis=0).flatten()
    all_targets_t1 = np.concatenate(all_targets_t1, axis=0).flatten()
    
    # Calculate Pearson correlations
    c0 = weighted_pearson_correlation(all_targets_t0, all_preds_t0)
    c1 = weighted_pearson_correlation(all_targets_t1, all_preds_t1)
    
    mean_corr = (c0 + c1) / 2
    
    return mean_corr, c0, c1


def train():
    """Main training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # add mps device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    set_seed(CFG.SEED)
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize W&B if available
    use_wandb = WANDB_API_KEY is not None
    if use_wandb:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=CFG.PROJECT_NAME, 
            name=CFG.EXPERIMENT_NAME,
            config={
                "input_dim": CFG.INPUT_DIM,
                "hidden_dim": CFG.HIDDEN_DIM,
                "num_layers": CFG.NUM_LAYERS,
                "dropout": CFG.DROPOUT,
                "batch_size": CFG.BATCH_SIZE,
                "learning_rate": CFG.LEARNING_RATE,
                "loss_weight_t0": CFG.LOSS_WEIGHT_T0,
                "loss_weight_t1": CFG.LOSS_WEIGHT_T1,
                "mse_reg_weight": CFG.MSE_REG_WEIGHT,
            }
        )
    
    # Load datasets
    train_dataset = LOBDataset(os.path.join(ROOT_DIR, CFG.TRAIN_PATH))
    valid_dataset = LOBDataset(os.path.join(ROOT_DIR, CFG.VALID_PATH))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    input_dim = len(train_dataset.feature_cols)
    print(f"\nModel Configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {CFG.HIDDEN_DIM}")
    print(f"  Num layers: {CFG.NUM_LAYERS}")
    print(f"  Head hidden: {CFG.HEAD_HIDDEN}")
    print(f"  Dropout t0: {CFG.DROPOUT_T0}")
    print(f"  Dropout t1: {CFG.DROPOUT_T1}")
    
    model = DualHeadGRU(
        input_dim=input_dim,
        hidden_dim=CFG.HIDDEN_DIM,
        num_layers=CFG.NUM_LAYERS,
        head_hidden=CFG.HEAD_HIDDEN,
        dropout=CFG.DROPOUT,
        dropout_t0=CFG.DROPOUT_T0,
        dropout_t1=CFG.DROPOUT_T1
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CFG.LEARNING_RATE, 
        weight_decay=CFG.WEIGHT_DECAY
    )
    
    # Use OneCycleLR with warmup - good for Pearson optimization
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=CFG.EPOCHS,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # Start LR = max_lr / 10
        final_div_factor=100  # End LR = max_lr / 100
    )
    
    # Training loop
    best_val_score = -1.0
    patience_counter = 0
    output_dir = os.path.join(ROOT_DIR, CFG.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting training with seed {CFG.SEED} for {CFG.EPOCHS} epochs...")
    print(f"Output directory: {output_dir}\n")
    
    for epoch in range(CFG.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_t0 = 0.0
        train_loss_t1 = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred_t0, pred_t1, _ = model(X)
            
            # Combined Pearson + MSE loss
            loss, loss_t0, loss_t1 = combined_loss(
                pred_t0.squeeze(-1), pred_t1.squeeze(-1),
                y[:, :, 0], y[:, :, 1],
                weight_t0=CFG.LOSS_WEIGHT_T0,
                weight_t1=CFG.LOSS_WEIGHT_T1,
                mse_reg=CFG.MSE_REG_WEIGHT
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRADIENT_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_loss_t0 += loss_t0.item()
            train_loss_t1 += loss_t1.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_loss_t0 = train_loss_t0 / len(train_loader)
        avg_loss_t1 = train_loss_t1 / len(train_loader)
        
        # Validation phase
        val_score, val_c0, val_c1 = validate(model, valid_loader, device)
        
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (t0: {avg_loss_t0:.4f}, t1: {avg_loss_t1:.4f})")
        print(f"  Val Pearson: {val_score:.4f} (t0: {val_c0:.4f}, t1: {val_c1:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_loss_t0": avg_loss_t0,
                "train_loss_t1": avg_loss_t1,
                "val_pearson": val_score,
                "val_pearson_t0": val_c0,
                "val_pearson_t1": val_c1,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (score: {best_val_score:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CFG.PATIENCE})")
        
        # Early stopping
        if patience_counter >= CFG.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print(f"\nTraining complete!")
    print(f"Best validation score: {best_val_score:.4f}")
    
    if use_wandb:
        wandb.finish()
    
    return best_val_score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train exp_09 model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        CFG.EPOCHS = args.epochs
    if args.batch_size is not None:
        CFG.BATCH_SIZE = args.batch_size
    
    train()
