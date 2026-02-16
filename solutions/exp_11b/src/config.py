"""
Configuration for exp_11b: Dual Causal Transformers for t0/t1

Two separate transformer models with different hyperparameters:
- Model T0: Simpler (t0 is easier to predict)
- Model T1: Deeper (t1 is harder, needs more capacity)
"""

# Model T0 Architecture (simpler - for easier target)
MODEL_T0_CONFIG = {
    'input_dim': 32,
    'd_model': 48,              # Smaller model dimension
    'n_heads': 4,
    'n_layers': 2,              # Fewer layers
    'd_ff': 96,
    'head_hidden': 32,
    'dropout': 0.2,             # Less regularization
}

# Model T1 Architecture (deeper - for harder target)  
MODEL_T1_CONFIG = {
    'input_dim': 32,
    'd_model': 96,              # Larger model dimension
    'n_heads': 6,
    'n_layers': 4,              # More layers for capacity
    'd_ff': 256,
    'head_hidden': 64,
    'dropout': 0.4,             # More regularization
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,               # More epochs - exp_11 showed slow improvement
    
    # Separate learning rates
    'lr_t0': 5e-4,              # Higher LR for simpler model
    'lr_t1': 2e-4,              # Lower LR for complex model
    
    'weight_decay': 1e-3,
    'gradient_clip': 0.5,
    
    # Loss weights (equal since separate models)
    't0_weight': 0.5,
    't1_weight': 0.5,
    
    # Huber loss
    'huber_delta': 1.0,
    
    # Early stopping
    'patience': 12,             # More patience for slow convergence
    'min_delta': 0.0005,
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
    'model_t0_save': 'solutions/exp_11b/best_model_t0.pth',
    'model_t1_save': 'solutions/exp_11b/best_model_t1.pth',
    'wandb_project': "WunderNN-Predictorium",
    'wandb_name': 'exp_11b_dual_transformers',
}
