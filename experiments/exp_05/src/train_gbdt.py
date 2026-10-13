import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config as CFG
from features import generate_features, get_feature_columns
import utils

def train_one_target(X, y, target_name):
    print(f"Training CatBoost for {target_name}...")
    # Train/Val Split?
    # We use validation set provided
    # But X, y here is Combined? No, we should load Train/Valid sets separately for CatBoost logic?
    # Or just pass simple Pools.
    pass

def flatten_dataset(parquet_path):
    print(f"Loading {parquet_path} for GBDT...")
    df = pd.read_parquet(parquet_path)
    df = generate_features(df)
    feature_cols = get_feature_columns(df)
    
    # Flatten: CatBoost handles rows.
    # Prediction: Steps 99..999 matter most, but we can train on all.
    # Or strict: Train on 0..999?
    X = df[feature_cols].values
    y = df[['t0', 't1']].values
    return X, y, feature_cols

def train_gbdt():
    # Load Train
    X_train, y_train, feat_names = flatten_dataset(CFG.TRAIN_PATH)
    # Load Valid
    X_val, y_val, _ = flatten_dataset(CFG.VALID_PATH)
    
    output_dir = os.path.join(CFG.ROOT_DIR, CFG.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    models = {}
    for i, target in enumerate(['t0', 't1']):
        print(f"--- Training Target {target} ---")
        train_pool = Pool(X_train, y_train[:, i], feature_names=feat_names)
        val_pool = Pool(X_val, y_val[:, i], feature_names=feat_names)
        
        model = CatBoostRegressor(
            iterations=CFG.GBDT_ITERATIONS,
            learning_rate=CFG.GBDT_LR,
            depth=CFG.GBDT_DEPTH,
            loss_function='RMSE',
            eval_metric='RMSE', # Or Pearson? CatBoost support custom metrics but RMSE is stable.
            verbose=100,
            allow_writing_files=False,
            task_type="CPU" # Or GPU
        )
        
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        
        model_path = os.path.join(output_dir, f"model_gbdt_{target}.cbm")
        model.save_model(model_path)
        print(f"Saved {model_path}")

        # Eval Pearson
        preds = model.predict(val_pool)
        # Score on FULL set (0..999)? Or metric?
        # Score on masked?
        # Let's create dummy mask? Or just all.
        # Mask provided in utils?
        # We can implement Weighted Pearson logic on flat array.
        # But we don't have step weights easily here (need step index?)
        # Approx score for log.
        score = np.corrcoef(y_val[:, i], preds)[0, 1]
        print(f"Val Pearson (Approx): {score:.4f}")

if __name__ == "__main__":
    train_gbdt()
