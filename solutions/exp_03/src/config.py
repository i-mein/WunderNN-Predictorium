from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    EXP_NAME = "exp_03" #"exp_03_hybrid"
    
    # Paths (Relative to project root)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # .../solutions/exp_03/src -> .../solutions/exp_03 -> .../solutions -> .../Project
    # Actually simpler to rely on CWD or explicitly pass
    # But for train.py execution, we usually run from root
    
    DATA_DIR = ROOT_DIR + "/datasets"
    TRAIN_PATH = DATA_DIR + "/train.parquet"
    VALID_PATH = DATA_DIR + "/valid.parquet"

    EXP_DIR = ROOT_DIR + "/solutions/" + EXP_NAME
    OUTPUT_DIR = EXP_DIR
    
    # Data
    # Base 49 + Rolling Features
    # Rolling on [mid_price_0, spread_0, vol_imbal_total]
    # Windows: [10, 50]
    # Stats: [mean, std]
    # 3 features * 2 windows * 2 stats = 12 new features.
    # Diff features (4) + Raw (32) + Eng (13) = 49.
    # Total = 49 + 12 = 61.
    INPUT_DIM = 61 
    OUTPUT_DIM = 2 # t0, t1

    # Model (Hybrid CNN-GRU)
    CNN_CHANNELS = 64
    KERNEL_SIZE = 3
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64 
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Misc
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_NAME = "WunderNN-Predictorium"

# check
cfg = Config()
print('ROOT_DIR', cfg.ROOT_DIR)
print('EXP_DIR', cfg.EXP_DIR)
print('DATA_DIR', cfg.DATA_DIR)
print('OUTPUT_DIR', cfg.OUTPUT_DIR)

