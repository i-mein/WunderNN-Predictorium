"""
exp_28: Training script for Advanced Transformer

Key features from aicheye/wundernn adapted for our competition:
- Regularized loss function (MSE + Huber + L1 + temporal consistency)
- Causal normalization (EMA-based, matches inference)
- OneCycleLR with warmup
- Early stopping with patience
- Weighted Pearson correlation for validation metric
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import AdvancedTransformerPredictor
from config import MODEL_CONFIG, TRAIN_CONFIG, get_model_kwargs
from src.utils import weighted_pearson_correlation

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")


# ============================================
# Loss Functions
# ============================================
class RegularizedLoss(nn.Module):
    """
    Composite loss function from aicheye/wundernn.
    
    Combines multiple loss terms for robust training:
    - MSE: Standard regression loss
    - Huber: Robust to outliers
    - L1: Sparse gradients for large errors
    - Weighted MSE: Emphasizes harder features
    - Temporal: Smoothness constraint
    - Reconstruction: Auxiliary regularization
    """
    
    def __init__(self, weights: dict, huber_delta: float = 0.3):
        super().__init__()
        self.weights = weights
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, outputs, targets, reconstruction=None, input_features=None):
        """
        Compute composite loss.
        
        Args:
            outputs: [batch, output_dim] predictions
            targets: [batch, output_dim] ground truth
            reconstruction: [batch, input_dim] reconstructed input (optional)
            input_features: [batch, input_dim] original input (optional)
            
        Returns:
            total_loss: Scalar loss value
            loss_dict: Dict with individual loss components
        """
        # Main losses
        mse = self.mse_loss(outputs, targets)
        huber = self.huber_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        
        # Feature-weighted MSE (emphasize harder features)
        feature_weights = torch.std(targets, dim=0) + 1e-8
        feature_weights = feature_weights / feature_weights.sum() * targets.size(1)
        weighted_mse = torch.mean(
            (outputs - targets) ** 2 * feature_weights.unsqueeze(0)
        )
        
        # Temporal consistency (smooth predictions over batch)
        if outputs.size(0) > 1:
            temporal_diff = torch.diff(outputs, dim=0)
            target_diff = torch.diff(targets, dim=0)
            temporal = self.mse_loss(temporal_diff, target_diff)
        else:
            temporal = torch.tensor(0.0, device=outputs.device)
        
        # Reconstruction loss
        if reconstruction is not None and input_features is not None:
            recon = self.mse_loss(reconstruction, input_features[:, -1])
        else:
            recon = torch.tensor(0.0, device=outputs.device)
        
        # Combine losses
        total_loss = (
            self.weights["mse"] * mse
            + self.weights["huber"] * huber
            + self.weights["l1"] * l1
            + self.weights["weighted_mse"] * weighted_mse
            + self.weights["temporal"] * temporal
            + self.weights["recon"] * recon
        )
        
        loss_dict = {
            "mse": mse.item(),
            "huber": huber.item(),
            "l1": l1.item(),
            "weighted_mse": weighted_mse.item(),
            "temporal": temporal.item() if isinstance(temporal, torch.Tensor) else 0.0,
            "recon": recon.item() if isinstance(recon, torch.Tensor) else 0.0,
        }
        
        return total_loss, loss_dict


class WeightedPearsonLoss(nn.Module):
    """
    Loss based on negative weighted Pearson correlation.
    
    This directly optimizes the competition metric.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute negative weighted Pearson correlation.
        
        Args:
            pred: [batch, 2] predictions for t0, t1
            target: [batch, 2] ground truth
            
        Returns:
            loss: 1 - weighted_corr (to minimize)
        """
        losses = []
        
        for i in range(pred.size(1)):
            pred_i = pred[:, i]
            target_i = target[:, i]
            
            # Weights based on target magnitude (competition metric)
            weights = torch.abs(target_i) + 1e-8
            
            # Weighted means
            sum_w = weights.sum()
            mean_pred = (pred_i * weights).sum() / sum_w
            mean_target = (target_i * weights).sum() / sum_w
            
            # Weighted deviations
            dev_pred = pred_i - mean_pred
            dev_target = target_i - mean_target
            
            # Weighted covariance and variances
            cov = (weights * dev_pred * dev_target).sum() / sum_w
            var_pred = (weights * dev_pred ** 2).sum() / sum_w
            var_target = (weights * dev_target ** 2).sum() / sum_w
            
            # Correlation
            corr = cov / (torch.sqrt(var_pred + 1e-8) * torch.sqrt(var_target + 1e-8))
            losses.append(1 - corr)
        
        # Weighted average: 2/3 * t0 + 1/3 * t1
        return (2/3) * losses[0] + (1/3) * losses[1]


# ============================================
# Dataset with Causal Normalization
# ============================================
class LOBSequenceDataset(Dataset):
    """
    Dataset for sequence-based training with causal normalization.
    
    Features:
    - Groups data by sequence ID
    - Applies EMA-based normalization (same as inference)
    - Returns context windows for transformer input
    """
    
    def __init__(
        self,
        parquet_path: str,
        seq_ids: list = None,
        context_len: int = 30,
        augment: bool = False,
        noise_std: float = 0.01,
        norm_alpha: float = 0.01,
    ):
        print(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.context_len = context_len
        self.augment = augment
        self.noise_std = noise_std
        self.norm_alpha = norm_alpha
        
        # Use provided seq_ids or all
        if seq_ids is not None:
            self.seq_ids = seq_ids
        else:
            self.seq_ids = self.df['seq_ix'].unique().tolist()
        
        # Feature columns (32 raw LOB features)
        self.feature_cols = [
            c for c in self.df.columns
            if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
        ]
        self.target_cols = ['t0', 't1']
        
        # Cache sequences with causal normalization applied
        print("Computing causal normalization and caching sequences...")
        self.data_cache = {}
        self._preprocess_sequences()
    
    def _causal_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply causal (online) normalization using EMA.
        
        This EXACTLY matches inference normalization to prevent leakage.
        """
        normalized = np.zeros_like(features)
        running_mean = None
        running_var = None
        alpha = self.norm_alpha
        
        for t in range(features.shape[0]):
            state = features[t]
            
            if running_mean is None:
                # Initialize with first observation
                running_mean = state.copy()
                running_var = np.ones_like(state, dtype=np.float32)
                # First timestep normalized to 0
                normalized[t] = np.zeros_like(state)
            else:
                # Normalize BEFORE update (causal)
                std = np.sqrt(running_var + 1e-8)
                normalized[t] = (state - running_mean) / std
            
            # Update running statistics
            delta = state - running_mean
            running_mean = running_mean + alpha * delta
            running_var = (1 - alpha) * running_var + alpha * (delta ** 2)
        
        # Clean up any NaNs/Infs
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def _preprocess_sequences(self):
        """Cache all sequences with causal normalization."""
        grouped = self.df.groupby('seq_ix')
        
        for seq_ix in tqdm(self.seq_ids, desc="Processing sequences"):
            group = grouped.get_group(seq_ix)
            
            # Raw features
            raw_features = group[self.feature_cols].values.astype(np.float32)
            
            # Apply causal normalization
            normalized = self._causal_normalize(raw_features)
            
            # Targets
            targets = group[self.target_cols].values.astype(np.float32)
            
            # need_prediction mask (for extracting valid training samples)
            need_pred = group['need_prediction'].values
            
            self.data_cache[seq_ix] = (normalized, targets, need_pred)
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sequence as context windows.
        
        Returns:
            contexts: [num_windows, context_len, input_dim]
            targets: [num_windows, output_dim]
        """
        seq_ix = self.seq_ids[idx]
        normalized, targets, need_pred = self.data_cache[seq_ix]
        
        # Extract windows where need_prediction=True
        contexts_list = []
        targets_list = []
        
        for t in range(len(normalized)):
            if not need_pred[t]:
                continue  # Skip warmup period
            
            # Get context window
            start_idx = max(0, t - self.context_len + 1)
            context = normalized[start_idx:t+1]
            
            # Pad if necessary
            if context.shape[0] < self.context_len:
                pad_len = self.context_len - context.shape[0]
                context = np.pad(
                    context, 
                    ((pad_len, 0), (0, 0)), 
                    mode='edge'  # Repeat first value for padding
                )
            
            # Optional augmentation (add noise)
            if self.augment and np.random.rand() < 0.5:
                noise = np.random.randn(*context.shape) * self.noise_std
                context = context + noise.astype(np.float32)
            
            contexts_list.append(context)
            targets_list.append(targets[t])
        
        if len(contexts_list) == 0:
            # Return dummy data if no valid samples (shouldn't happen)
            return (
                torch.zeros(1, self.context_len, len(self.feature_cols)),
                torch.zeros(1, len(self.target_cols)),
            )
        
        contexts = np.stack(contexts_list, axis=0)
        targets_arr = np.stack(targets_list, axis=0)
        
        return (
            torch.tensor(contexts, dtype=torch.float32),
            torch.tensor(targets_arr, dtype=torch.float32),
        )


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    
    Concatenates all windows from different sequences.
    """
    contexts = torch.cat([b[0] for b in batch], dim=0)
    targets = torch.cat([b[1] for b in batch], dim=0)
    return contexts, targets


# ============================================
# Training
# ============================================
def train():
    """Main training function."""
    
    # Initialize wandb
    if TRAIN_CONFIG["use_wandb"]:
        wandb.init(
            project=TRAIN_CONFIG["wandb_project"],
            name=TRAIN_CONFIG["experiment"],
            config={**MODEL_CONFIG, **TRAIN_CONFIG},
        )
    
    # Paths
    train_file = os.path.join(PROJECT_ROOT, "datasets", "train.parquet")
    valid_file = os.path.join(PROJECT_ROOT, "datasets", "valid.parquet")
    save_dir = os.path.dirname(SCRIPT_DIR)
    
    # Load training data (use ALL sequences for training)
    print("Loading data...")
    df_train = pd.read_parquet(train_file)
    train_seqs = df_train['seq_ix'].unique().tolist()
    
    # Load validation data from valid.parquet
    df_valid = pd.read_parquet(valid_file)
    val_seqs = df_valid['seq_ix'].unique().tolist()
    
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}")
    
    # Create datasets
    context_len = MODEL_CONFIG["context_len"]
    norm_alpha = TRAIN_CONFIG["norm_ema_alpha"]
    
    train_dataset = LOBSequenceDataset(
        train_file,
        seq_ids=train_seqs,
        context_len=context_len,
        augment=True,
        noise_std=TRAIN_CONFIG["noise_std"],
        norm_alpha=norm_alpha,
    )
    
    val_dataset = LOBSequenceDataset(
        valid_file,  # Use valid.parquet for validation
        seq_ids=val_seqs,
        context_len=context_len,
        augment=False,
        norm_alpha=norm_alpha,
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"],
    )
    
    # Initialize model
    print("Initializing model...")
    model = AdvancedTransformerPredictor(**get_model_kwargs(for_inference=False))
    model.to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler (OneCycleLR with warmup)
    total_steps = len(train_loader) * TRAIN_CONFIG["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_CONFIG["learning_rate"] * 4,
        total_steps=total_steps,
        pct_start=TRAIN_CONFIG["warmup_pct"],
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=1000,
    )
    
    # Loss function
    loss_fn = RegularizedLoss(
        weights=TRAIN_CONFIG["loss_weights"],
        huber_delta=TRAIN_CONFIG["huber_delta"],
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_corr = -float('inf')
    patience_counter = 0
    patience = TRAIN_CONFIG["patience"]
    
    print("Starting training...")
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        for contexts, targets in pbar:
            contexts = contexts.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(contexts, return_attention=False)
            predictions = output["prediction"]
            reconstruction = output["reconstruction"]
            
            # Compute loss
            loss, loss_dict = loss_fn(
                predictions, targets,
                reconstruction=reconstruction,
                input_features=contexts,
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
        
        train_loss /= train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for contexts, targets in val_loader:
                contexts = contexts.to(DEVICE)
                targets = targets.to(DEVICE)
                
                output = model(contexts, return_attention=False)
                predictions = output["prediction"]
                reconstruction = output["reconstruction"]
                
                loss, _ = loss_fn(
                    predictions, targets,
                    reconstruction=reconstruction,
                    input_features=contexts,
                )
                
                val_loss += loss.item()
                val_steps += 1
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        val_loss /= val_steps
        
        # Compute validation Pearson correlations
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        t0_corr = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
        t1_corr = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
        weighted_corr = (2/3) * t0_corr + (1/3) * t1_corr
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Val Corr: t0={t0_corr:.4f}, t1={t1_corr:.4f}, weighted={weighted_corr:.4f}")
        
        # Log to wandb
        if TRAIN_CONFIG["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_t0_corr": t0_corr,
                "val_t1_corr": t1_corr,
                "val_weighted_corr": weighted_corr,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        
        # Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  → Saved best model (loss={val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Also track best by correlation
        if weighted_corr > best_val_corr:
            best_val_corr = weighted_corr
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_corr.pth"))
            print(f"  → Saved best model by correlation (corr={weighted_corr:.4f})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final validation
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_model.pth"), map_location=DEVICE, weights_only=True)
    )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation correlation: {best_val_corr:.4f}")
    
    if TRAIN_CONFIG["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    train()
