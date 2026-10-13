import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import wandb
from dotenv import load_dotenv

# Adjust sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config as CFG
from model import CNNGRUModel
from features import generate_features, get_feature_columns
# Import from root src utils
PROJECT_ROOT = CFG.ROOT_DIR
sys.path.append(PROJECT_ROOT + "/src")
try:
    from utils import weighted_pearson_correlation
except ImportError:
    # Fallback if running relative
    sys.path.append("../../src")
    from utils import weighted_pearson_correlation

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

class FeaturedLOBDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print("Feature Engineering (Rolling Stats - this may take time)...")
        df = generate_features(df)
        
        self.feature_cols = get_feature_columns(df)
        print(f"Feature count: {len(self.feature_cols)}")
        
        if len(self.feature_cols) != CFG.INPUT_DIM:
            print(f"WARNING: Feature count {len(self.feature_cols)} != Config {CFG.INPUT_DIM}")
            
        self.target_cols = ['t0', 't1']
        self.seq_ids = df['seq_ix'].unique()
        self.grouped = df.groupby('seq_ix')
        self.data_cache = {}
        
        for seq_ix, group in tqdm(self.grouped, desc="Caching"):
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            self.data_cache[seq_ix] = (features, targets)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets = self.data_cache[seq_ix]
        return torch.from_numpy(features), torch.from_numpy(targets)

def train():
    use_wandb = WANDB_API_KEY is not None
    if use_wandb:
        wandb.login(key=WANDB_API_KEY)
        config_dict = {k: v for k, v in CFG.__dict__.items() if not k.startswith('__') and not callable(v)}
        wandb.init(project=CFG.PROJECT_NAME, name=CFG.EXP_NAME, config=config_dict)
    else:
        print("No WANDB key found.")

    train_dataset = FeaturedLOBDataset(CFG.TRAIN_PATH)
    valid_dataset = FeaturedLOBDataset(CFG.VALID_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Input Dim check
    real_input_dim = train_dataset.features.shape[1] if hasattr(train_dataset, 'features') else len(train_dataset.feature_cols)
    if real_input_dim != CFG.INPUT_DIM:
         print(f"Correction: Using detected Input Dim {real_input_dim}")
         
    model = CNNGRUModel(
        input_dim=real_input_dim,
        cnn_channels=CFG.CNN_CHANNELS,
        kernel_size=CFG.KERNEL_SIZE,
        hidden_dim=CFG.HIDDEN_DIM,
        num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT
    ).to(CFG.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=CFG.EPOCHS)
    
    best_val_score = -1.0
    output_dir = os.path.join(CFG.ROOT_DIR, CFG.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(CFG.EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        
        for X, y in pbar:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss/len(train_loader)
        val_score, val_loss = validate(model, valid_loader, criterion)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | Val Pearson {val_score:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch+1, 
                "train_loss": avg_train_loss, 
                "val_loss": val_loss, 
                "val_pearson": val_score,
                "lr": scheduler.get_last_lr()[0]
            })
            
        if val_score > best_val_score:
            best_val_score = val_score
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print("Saved Best Model.")
            
    if use_wandb: wandb.finish()

def validate(model, val_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
            outputs, _ = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            # Score on 99+
            all_preds.append(outputs[:, 99:, :].cpu().numpy())
            all_targets.append(y[:, 99:, :].cpu().numpy())
            
    val_loss /= len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    corr0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    corr1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    return (corr0+corr1)/2, val_loss

if __name__ == "__main__":
    train()
