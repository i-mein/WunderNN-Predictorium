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

# Adjust sys path to find src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config as CFG
from model import TransformerModel
from features import generate_features, get_feature_columns
# Import from root src utils for Pearson
# Assuming running from project root using .venv/bin/python solutions/exp_03/src/train.py
# The utils should be in solutions/exp_03/src/utils.py (we copied it)
from utils import weighted_pearson_correlation

# Load env
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

class FeaturedLOBDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print("Feature Engineering...")
        df = generate_features(df)
        
        self.feature_cols = get_feature_columns(df)
        # Ensure correct input dim 
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
    # WandB Init
    use_wandb = WANDB_API_KEY is not None
    if use_wandb:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=CFG.PROJECT_NAME, 
            name=CFG.EXP_NAME,
            config=CFG.__dict__
        )
    else:
        print("WANDB_API_KEY not found. Logging disabled.")

    # Data
    train_dataset = FeaturedLOBDataset(CFG.TRAIN_PATH)
    valid_dataset = FeaturedLOBDataset(CFG.VALID_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Model
    model = TransformerModel(
        input_dim=CFG.INPUT_DIM,
        d_model=CFG.D_MODEL,
        nhead=CFG.NHEAD,
        num_layers=CFG.NUM_LAYERS,
        dim_feedforward=CFG.DIM_FEEDFORWARD,
        dropout=CFG.DROPOUT
    ).to(CFG.DEVICE)
    
    # Optim
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=CFG.EPOCHS)
    
    best_val_score = -1.0
    
    for epoch in range(CFG.EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Train]")
        for X, y in pbar:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
            
            # Transformer needs causal mask
            # X shape: (Batch, Seq, Feat)
            # mask handled inside model forward
            
            optimizer.zero_grad()
            outputs, _ = model(X) # mask generated internally if None
            
            loss = criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss/len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_score, val_loss = validate(model, valid_loader, criterion)
        print(f"Epoch {epoch+1} Val Pearson: {val_score:.4f} | Val Loss: {val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_pearson": val_score,
                "lr": scheduler.get_last_lr()[0]
            })
        # Add best model saving 
        save_model_path = CFG.OUTPUT_DIR + f"/best_model.pth"    
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved new best model to {save_model_path}")
            
    if use_wandb:
        wandb.finish()

def validate(model, val_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validating"):
            X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
            outputs, _ = model(X) # Causal mask internal
            
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            # Predict for steps >= 99
            preds_slice = outputs[:, 99:, :]
            targets_slice = y[:, 99:, :]
            
            all_preds.append(preds_slice.cpu().numpy())
            all_targets.append(targets_slice.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    corr_t0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    corr_t1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    
    avg_val_loss = val_loss / len(val_loader)
    return (corr_t0 + corr_t1) / 2, avg_val_loss

if __name__ == "__main__":
    train()
