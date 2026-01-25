from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    EXP_NAME = "exp_05_ensemble"
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_05"
    
    # Feature Set:
    # Base (Exp 02/04) + New Exploration
    # Rough Count: ~100
    INPUT_DIM = 2 # Placeholder, calculate dynamically
    OUTPUT_DIM = 2 

    # DL Model (Attention-GRU)
    CNN_CHANNELS = 64
    KERNEL_SIZE = 3
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    ATTENTION_DIM = 64
    
    # Training DL
    BATCH_SIZE = 64 
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # GBDT
    GBDT_ITERATIONS = 500
    GBDT_DEPTH = 6
    GBDT_LR = 0.1
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_NAME = "WunderNN-Predictorium"
