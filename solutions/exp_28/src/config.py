"""
exp_28: Advanced Transformer for LOB Prediction - Configuration

Architecture from aicheye/wundernn adapted for:
- 2 regression targets (t0, t1) instead of state vector
- Weighted Pearson correlation metric
- Causal (online) normalization to prevent leakage
- Stateful inference with sequence history

Shared configuration between train.py and solution.py to ensure consistency.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Input/Output dimensions
    "input_dim": 32,        # 32 LOB features
    "output_dim": 2,        # 2 targets: t0, t1
    
    # Transformer architecture (from aicheye/wundernn)
    "d_model": 48,          # Embedding dimension (reduced from 64)
    "nhead": 6,             # Attention heads (reduced from 8)
    "num_layers": 3,        # Transformer layers (reduced from 4)
    "dim_feedforward": 128, # FFN dimension (reduced from 256)
    
    # Context window for inference
    "context_len": 100,     # Number of past timesteps for attention (full warmup)
    
    # Feature attention
    "feature_attn_hidden": 16,  # Hidden dim for feature attention
}

# Training Configuration
TRAIN_CONFIG = {
    "experiment": "exp_28_advanced_transformer",
    "dropout": 0.2,         # Increased dropout for regularization
    "batch_size": 8, #64,       # Batch size for sequence training
    "epochs": 30, #50,           # Training epochs
    "learning_rate": 1e-4,  # Learning rate (Adam)
    "weight_decay": 0.02,   # Weight decay for regularization
    "warmup_pct": 0.1,      # LR warmup percentage
    "warmup_steps": 99,     # First 100 steps (0-99) not scored
    "patience": 8,          # Early stopping patience
    "num_workers": 0,       # DataLoader workers (0 to avoid macOS shared memory issues)
    
    # Loss weighting (composite loss)
    "loss_weights": {
        "mse": 0.4,
        "huber": 0.3,
        "l1": 0.1,
        "weighted_mse": 0.1,
        "temporal": 0.05,
        "recon": 0.05,
    },
    "huber_delta": 0.3,     # Huber loss delta
    
    # Causal normalization (matches inference)
    "norm_ema_alpha": 0.01,  # EMA decay for running statistics
    "noise_std": 0.01,       # Training augmentation noise
    
    # Logging
    "use_wandb": True,
    "wandb_project": "WunderNN-Predictorium",
}

# Inference Configuration
INFERENCE_CONFIG = {
    "dropout": 0.0,
}


def get_model_kwargs(for_inference=False):
    """
    Get model keyword arguments for AdvancedTransformerPredictor instantiation.
    
    Args:
        for_inference: If True, sets dropout=0 for deterministic inference
        
    Returns:
        Dict with model configuration
    """
    kwargs = {
        "input_dim": MODEL_CONFIG["input_dim"],
        "output_dim": MODEL_CONFIG["output_dim"],
        "d_model": MODEL_CONFIG["d_model"],
        "nhead": MODEL_CONFIG["nhead"],
        "num_layers": MODEL_CONFIG["num_layers"],
        "dim_feedforward": MODEL_CONFIG["dim_feedforward"],
        "feature_attn_hidden": MODEL_CONFIG["feature_attn_hidden"],
        "dropout": INFERENCE_CONFIG["dropout"] if for_inference else TRAIN_CONFIG["dropout"],
    }
    return kwargs
