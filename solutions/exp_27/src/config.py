"""
exp_27: LOB ConvNeXt - Configuration

ConvNeXt-based architecture for LOB prediction:
1. Window size: 100 timesteps (full warmup period)  
2. Causal normalization: Online EMA matching inference
3. ConvNeXt blocks with depthwise separable convolutions
4. Input shape: (B, 12, 100, 3) -> (batch, levels, timesteps, channels)

Shared configuration between train.py and solution.py to ensure consistency.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Data reshaping
    "input_dim": 32,
    "num_levels": 12,       # 6 bid + 6 ask levels
    "num_channels": 3,      # price, volume, delta (in_channels)
    "window_size": 100,     # Full warmup window (first 100 steps not scored)
    
    # ConvNeXt architecture
    "dims": (64, 128, 256, 512),     # Channel dims for 4 stages
    "depths": (3, 3, 9, 3),          # Blocks per stage (ConvNeXt-T pattern)
    "kernel_sizes": ((3, 7), (3, 7), (3, 7), (3, 5)),  # (level, time) kernels
    "stem_stride": (2, 4),           # 12->6 levels, 100->25 time
    "downsample_strides": ((2, 2), (1, 2), (1, 2)),    # Progressive downsampling
    "drop_path_rate": 0.1,           # Stochastic depth
    
    # GRU
    "gru_hidden": 128,
    "gru_layers": 2,
}

# Training Configuration
TRAIN_CONFIG = {
    "experiment": "exp_27_lob_convnext",
    "dropout": 0.2,
    "batch_size": 4,        # Full sequence at a time
    "epochs": 50,           # More epochs for larger model
    "learning_rate": 3e-4,  # Slightly lower for stability
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
    Get model keyword arguments for LOBConvNeXt instantiation.
    
    Args:
        for_inference: If True, uses dropout=0.0. Otherwise uses training dropout.
    
    Returns:
        Dictionary of kwargs for LOBConvNeXt
    """
    dropout = INFERENCE_CONFIG["dropout"] if for_inference else TRAIN_CONFIG["dropout"]
    
    return {
        "in_channels": MODEL_CONFIG["num_channels"],
        "num_levels": MODEL_CONFIG["num_levels"],
        "window_size": MODEL_CONFIG["window_size"],
        "dims": MODEL_CONFIG["dims"],
        "depths": MODEL_CONFIG["depths"],
        "kernel_sizes": MODEL_CONFIG["kernel_sizes"],
        "stem_stride": MODEL_CONFIG["stem_stride"],
        "downsample_strides": MODEL_CONFIG["downsample_strides"],
        "drop_path_rate": MODEL_CONFIG["drop_path_rate"],
        "gru_hidden": MODEL_CONFIG["gru_hidden"],
        "gru_layers": MODEL_CONFIG["gru_layers"],
        "dropout": dropout,
    }


# Combined CONFIG for backward compatibility with training script
CONFIG = {
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
}
