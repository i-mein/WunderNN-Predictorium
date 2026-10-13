from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    EXP_NAME = "exp_04_swa"
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_04"
    
    # Data
    # Base Exp 02 (49) 
    # + RSI(14) on mid0, spread0 (2)
    # + BB%B(20) on mid0, spread0 (2)
    # Total = 49 + 4 = 53?
    # Let's count in features.py
    INPUT_DIM = 53
    OUTPUT_DIM = 2 

    # Model (Hybrid CNN-GRU)
    CNN_CHANNELS = 64
    KERNEL_SIZE = 3
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3 # High dropout for regularization
    
    # Training with SWA
    BATCH_SIZE = 64 
    EPOCHS = 15 # Increased for SWA
    SWA_START_EPOCH = 10
    LEARNING_RATE = 1e-3
    SWA_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_NAME = "WunderNN-Predictorium"
