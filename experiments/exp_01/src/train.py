import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model import GRUModel
from src.utils import weighted_pearson_correlation
import glob

# Constants
SEQ_LEN = 1000
WARMUP_STEPS = 99 # First 99 are warmup, training targets start at 99? 
# NOTE: The problem statement says "The first 99 steps (0-98) ... we won't score your predictions here."
# Evaluation is on steps 99 to 999.
# We should probably train on everything but only optimize loss on steps 99+ or use all for learning dynamics?
# Better to focus on what is scored. But causal nature means early mistakes hurt state.
# Let's train on full sequence but weight loss 0 for warmup if needed.
# For simplicity, standard MSE on full sequence is a good start.

BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Using device: {DEVICE}")

class LOBDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.seq_ids = self.df['seq_ix'].unique()
        self.param_cols = [c for c in self.df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        self.target_cols = ['t0', 't1']
        
        print(f"Found {len(self.seq_ids)} sequences.")
        
        # Pre-group by sequence for faster access
        # This might consume a lot of RAM. If so, we might need a more memory-efficient approach.
        # 10k sequences * 1000 steps * 37 floats * 4 bytes ~ 1.5GB RAM. Should be fine.
        self.grouped = self.df.groupby('seq_ix')
        self.data_cache = {}
        for seq_ix, group in tqdm(self.grouped, desc="Caching sequences"):
            features = group[self.param_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)
            # Need prediction mask?
            # mask = group['need_prediction'].values.astype(bool)
            self.data_cache[seq_ix] = (features, targets)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_ix = self.seq_ids[idx]
        features, targets = self.data_cache[seq_ix]
        return torch.from_numpy(features), torch.from_numpy(targets)

def train(model, train_loader, valid_loader, epochs=10):
    criterion = nn.MSELoss() # Can switch to custom weighted pearson loss later
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
            
            # Loss calculation
            # We can compute loss on all steps or just valid ones.
            # Let's compute on all for now to stabilize gradients on full sequence
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
    
    # Needs detailed handling to match the competition metric exactly
    # We should use only step_in_seq >= 99
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validating"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs, _ = model(X)
            
            # Only score steps >= 99 (index 99 to end)
            # Or reliance on 'need_prediction' if we loaded it.
            # Assuming standard structure of 1000 steps, index 99 is 100th step.
            
            # Slice output and targets
            # Actually, let's grab everything and filter in metric calculation to be safe or consistent
            # But the metric function expects flat arrays.
            
            # Let's approximate: take last 901 steps
            preds_slice = outputs[:, 99:, :]
            targets_slice = y[:, 99:, :]
            
            all_preds.append(preds_slice.cpu().numpy())
            all_targets.append(targets_slice.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 2)
    
    # Calculate score
    # t0
    corr_t0 = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    # t1
    corr_t1 = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    
    return (corr_t0 + corr_t1) / 2

if __name__ == "__main__":
    train_path = "datasets/train.parquet"
    valid_path = "datasets/valid.parquet"
    
    if not os.path.exists(train_path):
        print("Training data not found.")
        exit(1)
        
    train_dataset = LOBDataset(train_path)
    valid_dataset = LOBDataset(valid_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = GRUModel(input_dim=32, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=2).to(DEVICE)
    
    train(model, train_loader, valid_loader, epochs=EPOCHS)
