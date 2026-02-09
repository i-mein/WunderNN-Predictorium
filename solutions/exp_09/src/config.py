"""
Configuration for Exp 09: Dual-Head GRU with Pearson Loss

Key improvements over exp_06:
- Dual-head architecture (separate t0/t1 predictions)
- Pearson correlation loss (direct metric optimization)
- Same leakage-free features as exp_06 (65 features)
- Same efficient inference as exp_06

Key improvements over exp_07:
- No complex CNN/attention (fast inference)
- No additional features (proven leakage-free)
- Pure NumPy inference (no pandas per step)
"""

class Config:
    # Reproducibility
    SEED = 42
    
    # Data
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    
    # Model Architecture - Simple dual-head GRU
    INPUT_DIM = 65  # Same as exp_06
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Head-specific settings
    HEAD_HIDDEN = 128
    DROPOUT_T0 = 0.2  # Lower dropout for stronger predictor
    DROPOUT_T1 = 0.3  # Higher dropout for weaker predictor
    
    # Training
    BATCH_SIZE = 64  # Same as exp_06
    EPOCHS = 30
    LEARNING_RATE = 2e-3  # Increased from 1e-3 for faster initial learning
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Loss weights - t1 needs more focus (harder to predict)
    LOSS_WEIGHT_T0 = 0.3 #0.4  # Reduced (easier target)
    LOSS_WEIGHT_T1 = 0.7 #0.6  # Increased (harder target, lower correlation)
    MSE_REG_WEIGHT = 0.1 #0.05  # Reduced to prioritize Pearson optimization
    
    # Early stopping
    PATIENCE = 10
    
    # Output
    OUTPUT_DIR = "solutions/exp_09"
    MODEL_NAME = "best_model.pth"

    # WandB Logging
    PROJECT_NAME = "WunderNN-Predictorium"
    EXPERIMENT_NAME = "exp_09_dual_head_pearson"
    USE_WANDB = True  # Set to False to disable W&B logging
