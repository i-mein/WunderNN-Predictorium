"""
exp_25b: LOB Vision Transformer - Configuration

Shared configuration between train.py and solution.py to ensure consistency.
This pattern should be followed for all experiments.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Data reshaping
    "input_dim": 32,
    "num_levels": 12,       # 6 bid + 6 ask levels
    "num_channels": 3,      # price, volume, delta
    "window_size": 10,      # Temporal window for 2D representation
    
    # 2D Patch embedding
    "patch_size_h": 4,      # 12 / 4 = 3 patches vertically (levels)
    "patch_size_w": 2,      # 10 / 2 = 5 patches horizontally (time)
    
    # Transformer
    "embed_dim": 64,
    "num_transformer_layers": 2,
    "num_heads": 4,
    "ff_dim": 256,
    
    # GRU
    "gru_hidden": 128,
    "gru_layers": 2,
    
    # Pooling
    "pool_type": "mean",    # 'mean', 'max', or 'attention'
}

# Training Configuration
TRAIN_CONFIG = {
    "experiment": "exp_25b_lob_vit_snapshot",
    "dropout": 0.2,
    "batch_size": 1,        # Full sequence at a time
    "epochs": 10,           # 10 epochs
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_steps": 99,
    "num_snapshots": 4,     # Save 4 best checkpoints
    "use_wandb": True,
    "wandb_project": "WunderNN-Predictorium",
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
