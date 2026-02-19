"""
exp_26: LOB Vision Transformer - Configuration

Improvements over exp_25b:
1. Window size: 100 timesteps (full warmup period)
2. Causal normalization: Online EMA matching inference
3. Attention pooling: Learnable attention over level patches
4. Better patch sizes: 3x5 patches (4 level patches x 20 time patches)

Shared configuration between train.py and solution.py to ensure consistency.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Data reshaping
    "input_dim": 32,
    "num_levels": 12,       # 6 bid + 6 ask levels
    "num_channels": 3,      # price, volume, delta
    "window_size": 100,     # Full warmup window (first 100 steps not scored)
    
    # 2D Patch embedding
    # H=12 levels, W=100 timesteps
    # patch_size_h=3 -> 12/3 = 4 patches vertically (captures 3 levels each)
    # patch_size_w=5 -> 100/5 = 20 patches horizontally (captures 5 timesteps each)
    # Total patches = 4 * 20 = 80 patches
    "patch_size_h": 3,      # 3 levels per patch -> 4 level patches
    "patch_size_w": 5,      # 5 timesteps per patch -> 20 time patches
    
    # Transformer
    "embed_dim": 64,
    "num_transformer_layers": 3,    # Increased for larger context
    "num_heads": 4,
    "ff_dim": 256,
    
    # GRU
    "gru_hidden": 128,
    "gru_layers": 2,
    
    # Pooling
    "pool_type": "attention",    # Learnable attention pooling
}

# Training Configuration
TRAIN_CONFIG = {
    "experiment": "exp_26_lob_vit_3x5",
    "dropout": 0.2,
    "batch_size": 4,        # Full sequence at a time
    "epochs": 15,           # More epochs for larger model
    "learning_rate": 5e-4,  # Slightly lower for stability
    "weight_decay": 1e-4,
    "warmup_steps": 99,     # First 100 steps (0-99) used for warmup, not scored
    "num_snapshots": 4,     # Save 4 best checkpoints
    "use_wandb": True,
    "wandb_project": "WunderNN-Predictorium",
    
    # Causal normalization parameters
    "norm_ema_alpha": 0.01,  # EMA decay for running statistics
}

# Inference Configuration (dropout=0 for inference)
INFERENCE_CONFIG = {
    "dropout": 0.0,
}


def get_model_kwargs(for_inference=False):
    """
    Get model keyword arguments for LOBViTModel instantiation.
    
    Args:
        for_inference: If True, uses dropout=0.0. Otherwise uses training dropout.
    
    Returns:
        Dictionary of kwargs for LOBViTModel
    """
    dropout = INFERENCE_CONFIG["dropout"] if for_inference else TRAIN_CONFIG["dropout"]
    
    return {
        "input_dim": MODEL_CONFIG["input_dim"],
        "num_levels": MODEL_CONFIG["num_levels"],
        "num_channels": MODEL_CONFIG["num_channels"],
        "window_size": MODEL_CONFIG["window_size"],
        "patch_size_h": MODEL_CONFIG["patch_size_h"],
        "patch_size_w": MODEL_CONFIG["patch_size_w"],
        "embed_dim": MODEL_CONFIG["embed_dim"],
        "num_transformer_layers": MODEL_CONFIG["num_transformer_layers"],
        "num_heads": MODEL_CONFIG["num_heads"],
        "ff_dim": MODEL_CONFIG["ff_dim"],
        "gru_hidden": MODEL_CONFIG["gru_hidden"],
        "gru_layers": MODEL_CONFIG["gru_layers"],
        "dropout": dropout,
        "pool_type": MODEL_CONFIG["pool_type"],
    }


# Combined CONFIG for backward compatibility with training script
CONFIG = {
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
}
