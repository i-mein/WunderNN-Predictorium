"""
exp_23b: Multi-Stream TCN + GRU with Attention - Configuration

Improvements over exp_23:
1. Multi-input streams: separate TCN for each feature category (prices, volumes, deltas)
2. Attention-based fusion before regression heads
3. First 100 timesteps used as warmup (excluded from loss/metrics)
4. Snapshot ensemble with epoch-based checkpoints
5. Causal normalization matching inference

Shared configuration between train.py and solution.py to ensure consistency.
"""

# Feature categories for multi-stream processing
# Raw features (32 total):
#   - p0-p5: bid prices (6), p6-p11: ask prices (6) -> 12 price features
#   - v0-v5: bid volumes (6), v6-v11: ask volumes (6) -> 12 volume features  
#   - dp0-dp3: delta prices (4), dv0-dv3: delta volumes (4) -> 8 delta features
# Enhanced features (15 total) - computed from raw

FEATURE_CATEGORIES = {
    "prices": {
        "raw_indices": list(range(0, 12)),     # p0-p11 (12 features)
        "enhanced_indices": [32, 33, 34, 35],  # mid_price, micro_price, spread, relative_spread
    },
    "volumes": {
        "raw_indices": list(range(12, 24)),    # v0-v11 (12 features)
        "enhanced_indices": [36, 37, 38, 42],  # vol_imbal_0, vol_imbal_total, wpp, log_depth_ratio
    },
    "deltas": {
        "raw_indices": list(range(24, 32)),    # dp0-dp3, dv0-dv3 (8 features)
        "enhanced_indices": [39, 40, 41],      # ofi, price_momentum, vol_momentum
    },
    "depth": {
        "raw_indices": [],                      # No raw features
        "enhanced_indices": [43, 44, 45, 46],  # bid_depth, ask_depth, mid_micro_diff, norm_spread
    },
}

# Model Architecture Configuration
MODEL_CONFIG = {
    # Total input dimension
    "input_dim": 47,           # 32 raw + 15 enhanced
    
    # Per-stream dimensions (calculated from FEATURE_CATEGORIES)
    "price_dim": 16,           # 12 raw + 4 enhanced
    "volume_dim": 16,          # 12 raw + 4 enhanced
    "delta_dim": 11,           # 8 raw + 3 enhanced
    "depth_dim": 4,            # 0 raw + 4 enhanced
    
    # TCN settings per stream
    "tcn_hidden": 32,          # Hidden dim per stream (smaller, we have 4 streams)
    "tcn_layers": 4,           # Number of TCN layers
    "kernel_size": 3,          # TCN kernel size
    
    # Fusion settings
    "fusion_dim": 128,         # Dimension after concatenating streams
    
    # Attention settings
    "num_attention_heads": 4,  # Multi-head attention
    "attention_dropout": 0.1,
    
    # GRU settings
    "gru_hidden": 128,
    "gru_layers": 2,
    
    # Output
    "output_dim": 2,           # t0, t1
}

# Training Configuration
TRAIN_CONFIG = {
    "experiment": "exp_23b_multistream_tcn_attention",
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_steps": 100,       # First 100 steps excluded from loss (0-99)
    "num_snapshots": 4,        # Save 4 best checkpoints
    "use_wandb": True,
    "wandb_project": "WunderNN-Predictorium",
    
    # Causal normalization
    "norm_ema_alpha": 0.01,
    
    # Loss weighting
    "mse_weight": 0.7,
    "pearson_weight": 0.3,
}

# Inference Configuration
INFERENCE_CONFIG = {
    "dropout": 0.0,
}


def get_feature_indices(category: str) -> list:
    """Get all feature indices for a category (raw + enhanced)."""
    cat = FEATURE_CATEGORIES[category]
    return cat["raw_indices"] + cat["enhanced_indices"]


def get_model_kwargs(for_inference=False):
    """
    Get model keyword arguments for MultiStreamTCNGRUModel instantiation.
    
    Args:
        for_inference: If True, uses dropout=0.0.
    
    Returns:
        Dictionary of kwargs for model
    """
    dropout = INFERENCE_CONFIG["dropout"] if for_inference else TRAIN_CONFIG["dropout"]
    
    return {
        "input_dim": MODEL_CONFIG["input_dim"],
        "price_dim": MODEL_CONFIG["price_dim"],
        "volume_dim": MODEL_CONFIG["volume_dim"],
        "delta_dim": MODEL_CONFIG["delta_dim"],
        "depth_dim": MODEL_CONFIG["depth_dim"],
        "tcn_hidden": MODEL_CONFIG["tcn_hidden"],
        "tcn_layers": MODEL_CONFIG["tcn_layers"],
        "kernel_size": MODEL_CONFIG["kernel_size"],
        "fusion_dim": MODEL_CONFIG["fusion_dim"],
        "num_attention_heads": MODEL_CONFIG["num_attention_heads"],
        "attention_dropout": MODEL_CONFIG["attention_dropout"],
        "gru_hidden": MODEL_CONFIG["gru_hidden"],
        "gru_layers": MODEL_CONFIG["gru_layers"],
        "output_dim": MODEL_CONFIG["output_dim"],
        "dropout": dropout,
    }


# Combined CONFIG for backward compatibility
CONFIG = {
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
}
