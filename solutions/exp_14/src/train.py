"""
Training script for exp_14: CatBoost Baseline.

Trains separate CatBoost models for t0 and t1 targets.
Uses the same 65 leakage-free features as exp_06.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from catboost import CatBoostRegressor, Pool
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config as CFG
from features import generate_features, get_feature_columns

# Load environment
load_dotenv()

# Optional: W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_and_prepare_data(parquet_path):
    """Load parquet file and generate features."""
    full_path = os.path.join(CFG.ROOT_DIR, parquet_path)
    print(f"Loading {full_path}...")
    df = pd.read_parquet(full_path)
    
    print("Generating features...")
    df = generate_features(df)
    
    feature_cols = get_feature_columns(df)
    print(f"Feature count: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y_t0 = df['t0'].values.astype(np.float32)
    y_t1 = df['t1'].values.astype(np.float32)
    
    return X, y_t0, y_t1, feature_cols, df


def evaluate_predictions(y_true, y_pred, target_name):
    """Calculate Pearson correlation."""
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def train_catboost_model(X_train, y_train, X_val, y_val, 
                         feature_names, target_name, output_dir):
    """Train a CatBoost model for one target."""
    print(f"\n{'='*50}")
    print(f"Training CatBoost for {target_name}")
    print(f"{'='*50}")
    
    # Create pools
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    
    # Initialize model
    model = CatBoostRegressor(
        iterations=CFG.ITERATIONS,
        learning_rate=CFG.LEARNING_RATE,
        depth=CFG.DEPTH,
        l2_leaf_reg=CFG.L2_LEAF_REG,
        random_strength=CFG.RANDOM_STRENGTH,
        bagging_temperature=CFG.BAGGING_TEMPERATURE,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=CFG.SEED,
        verbose=CFG.VERBOSE,
        allow_writing_files=False,
        task_type=CFG.TASK_TYPE,
        early_stopping_rounds=CFG.EARLY_STOPPING_ROUNDS
    )
    
    # Train
    model.fit(train_pool, eval_set=val_pool)
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_pearson = evaluate_predictions(y_val, val_pred, target_name)
    print(f"\n{target_name} Validation Pearson: {val_pearson:.4f}")
    
    # Save model
    model_name = CFG.MODEL_T0 if target_name == 't0' else CFG.MODEL_T1
    model_path = os.path.join(output_dir, model_name)
    model.save_model(model_path)
    print(f"Saved model to {model_path}")
    
    # Feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features for {target_name}:")
    print(importance_df.head(10).to_string(index=False))
    
    return model, val_pearson, importance_df


def train_multi_seed(X_train, y_train, X_val, y_val, 
                     feature_names, target_name, output_dir, seeds=[42, 123, 456]):
    """Train with multiple seeds and return average Pearson."""
    from catboost import Pool
    
    pearson_scores = []
    models = []
    
    for seed in seeds:
        print(f"\n  Training {target_name} with seed={seed}...")
        
        train_pool = Pool(X_train, y_train, feature_names=feature_names)
        val_pool = Pool(X_val, y_val, feature_names=feature_names)
        
        model = CatBoostRegressor(
            iterations=CFG.ITERATIONS,
            learning_rate=CFG.LEARNING_RATE,
            depth=CFG.DEPTH,
            l2_leaf_reg=CFG.L2_LEAF_REG,
            random_strength=CFG.RANDOM_STRENGTH,
            bagging_temperature=CFG.BAGGING_TEMPERATURE,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=seed,
            verbose=CFG.VERBOSE,
            allow_writing_files=False,
            task_type=CFG.TASK_TYPE,
            early_stopping_rounds=CFG.EARLY_STOPPING_ROUNDS
        )
        
        model.fit(train_pool, eval_set=val_pool)
        
        val_pred = model.predict(X_val)
        pearson = evaluate_predictions(y_val, val_pred, target_name)
        pearson_scores.append(pearson)
        models.append(model)
        
        print(f"    Seed {seed} Pearson: {pearson:.4f}")
    
    avg_pearson = np.mean(pearson_scores)
    std_pearson = np.std(pearson_scores)
    
    print(f"\n{target_name} Multi-seed results:")
    print(f"  Scores: {[f'{p:.4f}' for p in pearson_scores]}")
    print(f"  Mean: {avg_pearson:.4f} ± {std_pearson:.4f}")
    
    # Return best model (or first seed=42 for reproducibility)
    best_model = models[0]  # seed=42
    
    # Save model
    model_name = CFG.MODEL_T0 if target_name == 't0' else CFG.MODEL_T1
    model_path = os.path.join(output_dir, model_name)
    best_model.save_model(model_path)
    print(f"  Saved model to {model_path}")
    
    # Feature importance
    importance = best_model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return avg_pearson, std_pearson, importance_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-seed', action='store_true', 
                        help='Train with 3 seeds and average results')
    args = parser.parse_args()
    
    print("="*60)
    print("Exp 14: CatBoost Baseline")
    print("="*60)
    
    # Set seed
    np.random.seed(CFG.SEED)
    
    # Create output directory
    output_dir = os.path.join(CFG.ROOT_DIR, CFG.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Multi-seed mode: {args.multi_seed}")
    
    # Initialize W&B
    use_wandb = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY")
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="WunderNN-Predictorium",
            name="exp_14_catboost_baseline",
            config={
                "model": "CatBoost",
                "iterations": CFG.ITERATIONS,
                "learning_rate": CFG.LEARNING_RATE,
                "depth": CFG.DEPTH,
                "l2_leaf_reg": CFG.L2_LEAF_REG,
            }
        )
    
    # Load data
    X_train, y_train_t0, y_train_t1, feature_names, df_train = \
        load_and_prepare_data(CFG.TRAIN_PATH)
    X_val, y_val_t0, y_val_t1, _, df_val = \
        load_and_prepare_data(CFG.VALID_PATH)
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Features: {len(feature_names)}")
    
    if args.multi_seed:
        # Multi-seed training for robust validation
        print("\n" + "="*50)
        print("Multi-seed training (3 seeds)")
        print("="*50)
        
        pearson_t0, std_t0, importance_t0 = train_multi_seed(
            X_train, y_train_t0, X_val, y_val_t0,
            feature_names, 't0', output_dir
        )
        
        pearson_t1, std_t1, importance_t1 = train_multi_seed(
            X_train, y_train_t1, X_val, y_val_t1,
            feature_names, 't1', output_dir
        )
        
        val_pearson = (pearson_t0 + pearson_t1) / 2
        
        print("\n" + "="*60)
        print("FINAL RESULTS (Multi-seed)")
        print("="*60)
        print(f"Val Pearson t0: {pearson_t0:.4f} ± {std_t0:.4f}")
        print(f"Val Pearson t1: {pearson_t1:.4f} ± {std_t1:.4f}")
        print(f"Val Pearson (avg): {val_pearson:.4f}")
        
    else:
        # Single-seed training
        model_t0, pearson_t0, importance_t0 = train_catboost_model(
            X_train, y_train_t0, X_val, y_val_t0,
            feature_names, 't0', output_dir
        )
        
        model_t1, pearson_t1, importance_t1 = train_catboost_model(
            X_train, y_train_t1, X_val, y_val_t1,
            feature_names, 't1', output_dir
        )
        
        val_pearson = (pearson_t0 + pearson_t1) / 2
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Val Pearson t0: {pearson_t0:.4f}")
        print(f"Val Pearson t1: {pearson_t1:.4f}")
        print(f"Val Pearson (avg): {val_pearson:.4f}")
    
    # Save feature importance
    importance_t0.to_csv(os.path.join(output_dir, 'feature_importance_t0.csv'), index=False)
    importance_t1.to_csv(os.path.join(output_dir, 'feature_importance_t1.csv'), index=False)
    
    # Log to W&B
    if use_wandb:
        wandb.log({
            "val_pearson_t0": pearson_t0,
            "val_pearson_t1": pearson_t1,
            "val_pearson": val_pearson,
        })
        wandb.finish()
    
    print("\nTraining complete!")
    return val_pearson


if __name__ == "__main__":
    main()
