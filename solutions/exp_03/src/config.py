from dataclasses import dataclass
import torch

@dataclass
class Config:
    EXP_NAME = "exp_03"
    # Paths
    ROOT_DIR = "/Users/ime/Work/WunderNN-Predictorium/"
    DATA_DIR = ROOT_DIR + "datasets"
    OUTPUT_DIR = ROOT_DIR + f"solutions/{EXP_NAME}"

    # Data
    TRAIN_PATH = DATA_DIR + "/train.parquet"
    VALID_PATH = DATA_DIR + "/valid.parquet"
    INPUT_DIM = 49 # From Exp 02 features
    OUTPUT_DIM = 2 # t0, t1

    # Model (Transformer)
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 64 # Increased form 32 for transformer stability
    EPOCHS = 10
    LEARNING_RATE = 1e-4 # Transformer needs smaller LR usually
    WEIGHT_DECAY = 1e-5
    
    # Misc
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_NAME = "WunderNN-Predictorium"
