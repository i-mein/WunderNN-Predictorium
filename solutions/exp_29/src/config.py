"""
exp_29: LOBViT-style 2D Transformer - Configuration

2D input reshaping: Raw 32 features → (B, 12, T, C)
- H=12: LOB levels (6 bid + 6 ask)
- W=T: Timesteps (context window)
- C=2: Channels (price, volume) - deltas computed temporally

Architecture:
- Patch embedding: (patch_h, patch_w) patches → tokens
- Positional encoding for patches
- Transformer encoder
- Regression head for t0, t1

Shared configuration between train.py and solution.py.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Input dimensions
    "input_dim": 32,        # Raw LOB features
    "output_dim": 2,        # Targets: t0, t1
    
    # 2D reshaping
    "num_levels": 12,       # 6 bid + 6 ask levels
    "num_channels": 2,      # price, volume (deltas computed from temporal diff)
    "context_len": 100,     # Timesteps in context window
    
    # Patch embedding (ViT-style)
    "patch_size": (3, 10),  # (levels, timesteps) → 4×10=40 patches
    "embed_dim": 64,        # Embedding dimension per patch
    
    # Transformer
    "num_heads": 4,         # Attention heads
    "num_layers": 4,        # Transformer layers
    "mlp_ratio": 4,         # FFN expansion ratio
    
    # Additional features
    "use_delta_channel": True,  # Add temporal diff as 3rd channel
    "use_cls_token": True,      # Use CLS token for pooling
}

# Training Configuration  
TRAIN_CONFIG = {
    "experiment": "exp_29_lobvit_2d",
    "dropout": 0.1,         # Dropout rate
    "drop_path": 0.1,       # Stochastic depth
    "batch_size": 8,        # Batch size (full sequences)
    "epochs": 30,           # Training epochs
    "learning_rate": 3e-4,  # Learning rate
    "weight_decay": 0.05,   # Weight decay
    "warmup_pct": 0.1,      # LR warmup percentage
    "patience": 10,         # Early stopping patience
    "num_workers": 0,       # DataLoader workers (0 for macOS)
    
    # Loss weighting
    "loss_weights": {
        "mse": 0.4,
        "huber": 0.3,
        "l1": 0.1,
        "weighted_mse": 0.1,
        "temporal": 0.05,
        "pearson": 0.05,
    },
    "huber_delta": 0.3,
    
    # Causal normalization
    "norm_ema_alpha": 0.01,
    "noise_std": 0.005,     # Smaller noise for 2D input
    
    # Logging
    "use_wandb": True,
    "wandb_project": "WunderNN-Predictorium",
}

# Inference Configuration
INFERENCE_CONFIG = {
    "dropout": 0.0,
}


def get_model_kwargs(for_inference=False):
    """Get model kwargs for LOBViT instantiation."""
    kwargs = {
        "num_levels": MODEL_CONFIG["num_levels"],
        "context_len": MODEL_CONFIG["context_len"],
        "num_channels": 3 if MODEL_CONFIG["use_delta_channel"] else MODEL_CONFIG["num_channels"],
        "patch_size": MODEL_CONFIG["patch_size"],
        "embed_dim": MODEL_CONFIG["embed_dim"],
        "num_heads": MODEL_CONFIG["num_heads"],
        "num_layers": MODEL_CONFIG["num_layers"],
        "mlp_ratio": MODEL_CONFIG["mlp_ratio"],
        "output_dim": MODEL_CONFIG["output_dim"],
        "use_cls_token": MODEL_CONFIG["use_cls_token"],
        "dropout": INFERENCE_CONFIG["dropout"] if for_inference else TRAIN_CONFIG["dropout"],
        "drop_path": 0.0 if for_inference else TRAIN_CONFIG["drop_path"],
    }
    return kwargs
