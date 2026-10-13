import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from solutions.exp_02.model import GRUModel
from solutions.exp_02.features import generate_features, get_feature_columns
import sys

# Adjust utils import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")
from src.utils import weighted_pearson_correlation

# CONFIG 
BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Using device: {DEVICE}")



class FeaturedLOBDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading and processing data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Apply Feature Engineering
        print("Generating features...")
        df = generate_features(df)
        
        self.feature_cols = get_feature_columns(df)
        self.target_cols = ['t0', 't1']
        self.input_dim = len(self.feature_cols)
        
        print(f"Feature count: {self.input_dim}")
        print(f"Features: {self.feature_cols}")
        
        self.seq_ids = df['seq_ix'].unique()
        print(f"Found {len(self.seq_ids)} sequences.")
        
        self.grouped = df.groupby('seq_ix')
        self.data_cache = {}
        
        for seq_ix, group in tqdm(self.grouped, desc="Caching sequences"):
            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            self.data_cache[seq_ix] = (features, targets)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets = self.data_cache[seq_ix]
        return torch.from_numpy(features), torch.from_numpy(targets)

def train(model, train_loader, valid_loader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=epochs)
    
    best_val_score = -1.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(X)
            loss = criterion(outputs, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        val_score = validate(model, valid_loader)
        print(f"Epoch {epoch+1} Val Weighted Pearson: {val_score:.4f}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model.")

def validate(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validating"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs, _ = model(X)
            
            # Predict for steps >= 99
            preds_slice = outputs[:, 99:, :]
            targets_slice = y[:, 99:, :]
            
            all_preds.append(preds_slice.cpu().numpy())
            all_targets.append(targets_slice.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    corr_t0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    corr_t1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    
    return (corr_t0 + corr_t1) / 2

if __name__ == "__main__":
    train_path = "datasets/train.parquet"
    valid_path = "datasets/valid.parquet"
    
    train_dataset = FeaturedLOBDataset(train_path)
    valid_dataset = FeaturedLOBDataset(valid_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = train_dataset.input_dim
    print(f"Model Input Dimension: {input_dim}")
    
    model = GRUModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=2).to(DEVICE)
    
    train(model, train_loader, valid_loader, epochs=EPOCHS)
