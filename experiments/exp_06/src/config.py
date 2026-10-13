from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    """Configuration for exp_06: Leakage-Free Feature Engineering."""
    
    EXP_NAME = "exp_06_leakage_free"
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_06"
    
    # Model Architecture
    INPUT_DIM = 65  # Will be calculated dynamically from features
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    OUTPUT_DIM = 2
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64  # Match exp_01/02
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Early stopping
    PATIENCE = 7
    
    # Loss weights (based on exp_01 correlations: t0=0.3342, t1=0.1277)
    LOSS_WEIGHT_T0 = 0.7
    LOSS_WEIGHT_T1 = 0.3
    
    # Device
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # Logging
    PROJECT_NAME = "WunderNN-Predictorium"
    USE_WANDB = True  # Set to False to disable W&B logging
