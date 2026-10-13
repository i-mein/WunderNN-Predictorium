"""
Configuration for exp_12: Specialized t1 Prediction Model

Focused on improving t1 prediction (the harder target).
"""

# Model Architecture
MODEL_CONFIG = {
    'input_dim': 32,            # Raw features only
    
    # GRU backbone (same as exp_01)
    'gru_hidden': 256,
    'gru_layers': 2,
    'gru_dropout': 0.3,
    
    # t0 head (simple, like exp_01)
    't0_hidden': 64,
    't0_dropout': 0.2,
    
    # t1 head (deeper, more capacity for harder task)
    't1_hidden1': 128,          # First hidden layer
    't1_hidden2': 64,           # Second hidden layer
    't1_dropout': 0.4,          # Higher dropout
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 25,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,
    'gradient_clip': 0.5,
    
    # Loss weights - FOCUS ON t1
    't0_weight': 0.4,           # Less weight on t0 (already good)
    't1_weight': 0.6,           # More weight on t1 (needs improvement)
    
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
    'model_save': 'solutions/exp_12/best_model.pth',
    'wandb_project': "WunderNN-Predictorium",
    'wandb_name': 'exp_12_t1_specialized',
}
