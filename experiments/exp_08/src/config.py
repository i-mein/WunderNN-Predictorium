from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    """Configuration for exp_07: Dual-Head Dual-Stream Architecture."""
    
    EXP_NAME = "exp_08_dual_head_combo_loss"
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_08"
    
    # Feature dimensions
    RAW_FEATURES = 32  # Original LOB features
    ENG_FEATURES = 46  # Engineered features (calculated dynamically)
    TOTAL_FEATURES = 78  # Actual total (32 raw + 46 engineered)
    
    # CNN Stream (for raw features)
    CNN_CHANNELS = [64, 128, 128]
    CNN_KERNELS = [3, 5, 7]
    
    # GRU Stream (for engineered features)
    GRU_HIDDEN = 256
    GRU_LAYERS = 2
    
    # Fusion
    FUSION_HEADS = 8
    
    # Task Heads
    HEAD_HIDDEN = 128
    HEAD_ATTENTION_HEADS = 4
    
    # Regularization
    DROPOUT_T0 = 0.2  # Lower dropout for stronger predictor
    DROPOUT_T1 = 0.3  # Higher dropout for weaker predictor
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Training
    BATCH_SIZE = 24 #48  # Smaller due to larger model
    EPOCHS = 30
    LEARNING_RATE = 5e-4  # Lower LR for stability
    
    # Early stopping
    PATIENCE = 10  # More patience for complex model
    
    # Loss weights
    LOSS_WEIGHT_T0 = 0.7
    LOSS_WEIGHT_T1 = 0.3
    MSE_REG_WEIGHT = 0.5 #0.1  # MSE regularization for t1
    
    # Device
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    # Seeds
    SEEDS = [42]
    
    # Logging
    PROJECT_NAME = "WunderNN-Predictorium"
    USE_WANDB = True  # Set to False to disable W&B logging
