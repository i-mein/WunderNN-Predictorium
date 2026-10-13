"""
Configuration for exp_10b: DeepLOB-style CNN-GRU with CAUSAL Convolutions

Based on:
- DeepLOB (Zhang et al., 2019) - arXiv:1808.03668
- Adapted for regression with dual heads (t0, t1)

exp_10b FIX: Uses causal convolutions (left-only padding) to prevent future leakage.
"""

# Model Architecture
MODEL_CONFIG = {
    'input_dim': 32,            # Raw features only (like exp_01)
    
    # Inception CNN modules
    'inception1_filters': 16,    # Filters per branch in inception 1
    'inception2_filters': 16,    # Filters per branch in inception 2
    'kernel_sizes': [1, 3, 5],   # Multi-scale kernels
    
    # GRU backbone
    'gru_hidden': 128,           # Hidden size (smaller than exp_06's 256)
    'gru_layers': 2,
    
    # Output heads
    'head_hidden': 64,
    
    # Regularization
    'dropout_cnn': 0.2,          # Dropout after CNN
    'dropout_gru': 0.4,          # Dropout in GRU
    'dropout_t0': 0.3,           # Dropout for t0 head
    'dropout_t1': 0.5,           # Higher dropout for t1 (harder target)
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32, #64,
    'epochs': 20, #30,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,        # Strong L2 regularization
    'gradient_clip': 0.5,
    
    # Loss weights
    't0_weight': 0.6,
    't1_weight': 0.4,
    
    # Huber loss (more robust than MSE)
    'loss_type': 'huber',
    'huber_delta': 1.0,
    
    # Early stopping
    'patience': 8,
    'min_delta': 0.001,
    
    # Data augmentation
    'input_noise_std': 0.01,     # Add small noise to inputs
    'feature_dropout': 0.05,     # Randomly zero features
}

# Data Configuration
DATA_CONFIG = {
    'train_path': 'datasets/train.parquet',
    'valid_path': 'datasets/valid.parquet',
    'sequence_length': 1000,
    
    # Use raw features only (32 features like exp_01)
    'use_raw_features_only': True,
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
    'model_save': 'solutions/exp_10b/best_model.pth',
    'wandb_project': "WunderNN-Predictorium",
    'wandb_name': 'exp_10b_deeplob_causal',
}
