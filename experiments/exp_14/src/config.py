"""
Configuration for Exp 14: CatBoost Baseline

Key approach:
- Use proven leakage-free features from exp_06 (65 features)
- Separate CatBoost models for t0 and t1
- Simple, fast baseline for GBDT path
"""
import os

class Config:
    # Reproducibility
    SEED = 42
    
    # Paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_14"
    
    # CatBoost hyperparameters
    ITERATIONS = 2000
    LEARNING_RATE = 0.03
    DEPTH = 6
    L2_LEAF_REG = 3.0
    RANDOM_STRENGTH = 1.0
    BAGGING_TEMPERATURE = 0.5
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 100
    
    # Task type
    TASK_TYPE = "CPU"  # or "GPU" if available
    
    # Logging
    VERBOSE = 100
    
    # Model names
    MODEL_T0 = "model_catboost_t0.cbm"
    MODEL_T1 = "model_catboost_t1.cbm"
