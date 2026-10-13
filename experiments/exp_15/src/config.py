"""
Configuration for exp_15: Enhanced CatBoost with advanced features.

Based on exp_14 but with additional micro-price and trade flow features.
"""
import os


class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    TRAIN_PATH = "datasets/train.parquet"
    VALID_PATH = "datasets/valid.parquet"
    OUTPUT_DIR = "solutions/exp_15"
    
    # Model names
    MODEL_T0 = "model_catboost_t0.cbm"
    MODEL_T1 = "model_catboost_t1.cbm"
    
    # CatBoost hyperparameters (same as exp_14 baseline)
    ITERATIONS = 3000
    LEARNING_RATE = 0.03
    DEPTH = 8
    L2_LEAF_REG = 3.0
    RANDOM_STRENGTH = 0.5
    BAGGING_TEMPERATURE = 0.0
    
    # Training settings
    SEED = 42
    SEEDS = [42, 123, 456]  # For multi-seed training
    VERBOSE = 100
    TASK_TYPE = 'CPU'
    EARLY_STOPPING_ROUNDS = 200
