import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config as CFG
from model import DualHeadDualStream
from features import generate_features, get_feature_columns, get_raw_feature_columns, get_engineered_feature_columns
from losses import combined_loss, pearson_correlation_loss
from utils import weighted_pearson_correlation

# Load environment variables
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

class LOBDataset(Dataset):
    """Dataset for Limit Order Book sequences with dual-stream features."""
    
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print("Generating features...")
        df = generate_features(df)
        
        self.all_feature_cols = get_feature_columns(df)
        self.raw_feature_cols = get_raw_feature_columns()
        self.eng_feature_cols = get_engineered_feature_columns(df)
        
        print(f"Total features: {len(self.all_feature_cols)}")
        print(f"Raw features: {len(self.raw_feature_cols)}")
        print(f"Engineered features: {len(self.eng_feature_cols)}")
        
        self.target_cols = ['t0', 't1']
        self.seq_ids = df['seq_ix'].unique()
        
        # Cache sequences for faster loading
        self.data_cache = {}
        grouped = df.groupby('seq_ix')
        
        for seq_ix, group in tqdm(grouped, desc="Caching sequences"):
            raw_features = group[self.raw_feature_cols].values.astype(np.float32)
            eng_features = group[self.eng_feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            self.data_cache[seq_ix] = (raw_features, eng_features, targets)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        raw_features, eng_features, targets = self.data_cache[seq_ix]
        return (
            torch.from_numpy(raw_features),
            torch.from_numpy(eng_features),
            torch.from_numpy(targets)
        )


def validate(model, val_loader, device):
    """
    CRITICAL: Leakage-free validation.
    Evaluates ALL timesteps (no warm-up period).
    Returns separate correlations for t0 and t1.
    """
    model.eval()
    all_preds_t0 = []
    all_preds_t1 = []
    all_targets = []
    
    with torch.no_grad():
        for raw_X, eng_X, y in val_loader:
            raw_X = raw_X.to(device)
            eng_X = eng_X.to(device)
            y = y.to(device)
            
            # Forward pass
            pred_t0, pred_t1, _, _, _ = model(raw_X, eng_X)
            
            # CRITICAL: Evaluate ALL timesteps
            all_preds_t0.append(pred_t0.cpu().numpy())
            all_preds_t1.append(pred_t1.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Flatten all predictions and targets
    all_preds_t0 = np.concatenate(all_preds_t0, axis=0).reshape(-1)
    all_preds_t1 = np.concatenate(all_preds_t1, axis=0).reshape(-1)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    # Calculate weighted Pearson correlation for each target
    c0 = weighted_pearson_correlation(all_targets[:, 0], all_preds_t0)
    c1 = weighted_pearson_correlation(all_targets[:, 1], all_preds_t1)
    
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
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    raw_dim = len(train_dataset.raw_feature_cols)
    eng_dim = len(train_dataset.eng_feature_cols)
    
    print(f"\nModel Configuration:")
    print(f"  Raw features: {raw_dim}")
    print(f"  Engineered features: {eng_dim}")
    print(f"  Total features: {raw_dim + eng_dim}")
    print(f"  CNN channels: {CFG.CNN_CHANNELS}")
    print(f"  GRU hidden: {CFG.GRU_HIDDEN}")
    print(f"  Fusion heads: {CFG.FUSION_HEADS}")
    print(f"  Device: {CFG.DEVICE}")
    
    model = DualHeadDualStream(
        raw_dim=raw_dim,
        eng_dim=eng_dim,
        cnn_channels=CFG.CNN_CHANNELS,
        cnn_kernels=CFG.CNN_KERNELS,
        gru_hidden=CFG.GRU_HIDDEN,
        gru_layers=CFG.GRU_LAYERS,
        fusion_heads=CFG.FUSION_HEADS,
        head_hidden=CFG.HEAD_HIDDEN,
        head_attention_heads=CFG.HEAD_ATTENTION_HEADS,
        dropout_t0=CFG.DROPOUT_T0,
        dropout_t1=CFG.DROPOUT_T1
    ).to(CFG.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
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
        train_loss_t0 = 0.0
        train_loss_t1 = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        for raw_X, eng_X, y in pbar:
            raw_X = raw_X.to(CFG.DEVICE)
            eng_X = eng_X.to(CFG.DEVICE)
            y = y.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_t0, pred_t1, _, _, _ = model(raw_X, eng_X)
            
            # Combined loss
            loss, loss_t0, loss_t1 = combined_loss(
                pred_t0, pred_t1,
                y[:, :, 0:1], y[:, :, 1:2],
                CFG.LOSS_WEIGHT_T0,
                CFG.LOSS_WEIGHT_T1,
                CFG.MSE_REG_WEIGHT
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRADIENT_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_loss_t0 += loss_t0.item()
            train_loss_t1 += loss_t1.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                't0': f'{loss_t0.item():.4f}',
                't1': f'{loss_t1.item():.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_t0 = train_loss_t0 / len(train_loader)
        avg_train_loss_t1 = train_loss_t1 / len(train_loader)
        
        # Validation phase
        val_score, val_c0, val_c1 = validate(model, valid_loader, CFG.DEVICE)
        
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (t0: {avg_train_loss_t0:.4f}, t1: {avg_train_loss_t1:.4f})")
        print(f"  Val Pearson: {val_score:.4f} (t0: {val_c0:.4f}, t1: {val_c1:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_loss_t0": avg_train_loss_t0,
                "train_loss_t1": avg_train_loss_t1,
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
    
    parser = argparse.ArgumentParser(description="Train exp_07 model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        CFG.EPOCHS = args.epochs
    if args.batch_size is not None:
        CFG.BATCH_SIZE = args.batch_size
    
    train()
