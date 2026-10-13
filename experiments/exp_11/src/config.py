"""
Configuration for exp_11: Causal Transformer for LOB Prediction

Based on "Transformers for Limit Order Books" (arXiv:2003.00130)
"""

# Model Architecture
MODEL_CONFIG = {
    'input_dim': 32,            # Raw features only
    
    # Transformer config
    'd_model': 64,              # Model dimension
    'n_heads': 4,               # Attention heads
    'n_layers': 2,              # Transformer layers
    'd_ff': 128,                # Feed-forward dimension
    
    # Output heads
    'head_hidden': 64,
    
    # Regularization
    'dropout': 0.3,
    'dropout_t0': 0.2,
    'dropout_t1': 0.4,          # More dropout for harder target
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 25,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,
    'gradient_clip': 0.5,
    
    # Loss weights
    't0_weight': 0.5,
    't1_weight': 0.5,
    
    # Huber loss
    'loss_type': 'huber',
    'huber_delta': 1.0,
    
    # Early stopping
    'patience': 8,
    'min_delta': 0.001,
}

# Data Configuration
DATA_CONFIG = {
    'train_path': 'datasets/train.parquet',
    'valid_path': 'datasets/valid.parquet',
    'sequence_length': 1000,
    
    'features': [
        'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11',
        'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
        'dp0', 'dp1', 'dp2', 'dp3',
        'dv0', 'dv1', 'dv2', 'dv3',
    ],
    'targets': ['t0', 't1'],
}

# Paths
PATHS = {
    'model_save': 'solutions/exp_11/best_model.pth',
    'wandb_project': "WunderNN-Predictorium",
    'wandb_name': 'exp_11_causal_transformer',
}
