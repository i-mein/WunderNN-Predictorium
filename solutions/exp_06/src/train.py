import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config as CFG
from model import UnidirectionalGRU
from features import generate_features, get_feature_columns
from utils import weighted_pearson_correlation

# Load environment variables
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

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


def weighted_mse_loss(predictions, targets, weight_t0=0.7, weight_t1=0.3):
    """
    Weighted MSE loss giving more importance to t0 (which has higher correlation).
    """
    mse_t0 = nn.functional.mse_loss(predictions[:, :, 0], targets[:, :, 0])
    mse_t1 = nn.functional.mse_loss(predictions[:, :, 1], targets[:, :, 1])
    return weight_t0 * mse_t0 + weight_t1 * mse_t1


def validate(model, val_loader, device):
    """
    CRITICAL: Leakage-free validation.
    Evaluates ALL timesteps (no warm-up period).
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs, _ = model(X)
            
            # CRITICAL: Evaluate ALL timesteps, not just 99+
            # This is the key difference from exp_03-05 which had leakage
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Flatten all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    # Calculate weighted Pearson correlation for each target
    c0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    c1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    
    mean_corr = (c0 + c1) / 2
    
    return mean_corr, c0, c1


def train():
    """Main training function."""
    
    # Initialize W&B if available
    use_wandb = WANDB_API_KEY and CFG.USE_WANDB
    if use_wandb:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        # Create config dict from class attributes
        config_dict = {k: v for k, v in vars(CFG).items() if not k.startswith('_')}
        wandb.init(project=CFG.PROJECT_NAME, name=CFG.EXP_NAME, config=config_dict)
    
    # Load datasets
    train_dataset = LOBDataset(os.path.join(CFG.ROOT_DIR, CFG.TRAIN_PATH))
    valid_dataset = LOBDataset(os.path.join(CFG.ROOT_DIR, CFG.VALID_PATH))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for speed
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
    print(f"  Dropout: {CFG.DROPOUT}")
    print(f"  Device: {CFG.DEVICE}")
    
    model = UnidirectionalGRU(
        input_dim=input_dim,
        hidden_dim=CFG.HIDDEN_DIM,
        num_layers=CFG.NUM_LAYERS,
        output_dim=CFG.OUTPUT_DIM,
        dropout=CFG.DROPOUT
    ).to(CFG.DEVICE)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CFG.LEARNING_RATE, 
        weight_decay=CFG.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=CFG.EPOCHS
    )
    
    # Training loop
    best_val_score = -1.0
    patience_counter = 0
    output_dir = os.path.join(CFG.ROOT_DIR, CFG.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nStarting training for {CFG.EPOCHS} epochs...")
    print(f"Output directory: {output_dir}\n")
    
    for epoch in range(CFG.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        for X, y in pbar:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(X)
            
            # Weighted MSE loss
            loss = weighted_mse_loss(
                outputs, y, 
                CFG.LOSS_WEIGHT_T0, 
                CFG.LOSS_WEIGHT_T1
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRADIENT_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_score, val_c0, val_c1 = validate(model, valid_loader, CFG.DEVICE)
        
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Pearson: {val_score:.4f} (t0: {val_c0:.4f}, t1: {val_c1:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
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
    
    parser = argparse.ArgumentParser(description="Train exp_06 model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        CFG.EPOCHS = args.epochs
    if args.batch_size is not None:
        CFG.BATCH_SIZE = args.batch_size
    
    train()
