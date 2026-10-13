"""
Configuration for exp_16: CatBoost with temporal/momentum features + Optuna.

Focus on improving t1 with longer-horizon features and separate hyperparameters.
"""
import os


class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_16"
    
    # Model names
    MODEL_T0 = "model_catboost_t0.cbm"
    MODEL_T1 = "model_catboost_t1.cbm"
    
    # Default CatBoost hyperparameters (will be overridden by Optuna)
    # t0 defaults (from exp_15)
    T0_ITERATIONS = 3000
    T0_LEARNING_RATE = 0.03
    T0_DEPTH = 8
    T0_L2_LEAF_REG = 3.0
    T0_RANDOM_STRENGTH = 0.5
    T0_BAGGING_TEMPERATURE = 0.0
    
    # t1 defaults (more conservative - to prevent overfitting)
    T1_ITERATIONS = 1000
    T1_LEARNING_RATE = 0.01
    T1_DEPTH = 6
    T1_L2_LEAF_REG = 5.0
    T1_RANDOM_STRENGTH = 1.0
    T1_BAGGING_TEMPERATURE = 0.5
    
    # Training settings
    SEED = 42
    VERBOSE = 100
    TASK_TYPE = 'CPU'
    EARLY_STOPPING_ROUNDS = 200
    
    # Optuna settings
    N_TRIALS = 30
    TIMEOUT = 3600  # 1 hour per target
