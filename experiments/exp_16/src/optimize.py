"""
Optuna hyperparameter optimization for exp_16.

Features:
- Separate optimization for t0 and t1
- Single seed (for speed)
- Focus on preventing t1 overfitting
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from catboost import CatBoostRegressor, Pool
import optuna
from optuna.samplers import TPESampler
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
    
    return X, y_t0, y_t1, feature_cols


def evaluate_predictions(y_true, y_pred):
    """Calculate Pearson correlation."""
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def create_objective(X_train, y_train, X_val, y_val, feature_names, target_name):
    """Create Optuna objective for a specific target."""
    
    def objective(trial):
        # Hyperparameter search space
        if target_name == 't0':
            # t0 can handle more complexity
            params = {
                'iterations': trial.suggest_int('iterations', 2000, 5000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'depth': trial.suggest_int('depth', 6, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            }
        else:
            # t1 needs more regularization to prevent overfitting
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 20.0),
                'random_strength': trial.suggest_float('random_strength', 0.5, 3.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.3, 1.5),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            }
        
        # Create pools
        train_pool = Pool(X_train, y_train, feature_names=feature_names)
        val_pool = Pool(X_val, y_val, feature_names=feature_names)
        
        # Common params
        model_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': CFG.SEED,
            'verbose': 0,
            'allow_writing_files': False,
            'task_type': 'CPU',
            'early_stopping_rounds': 100 if target_name == 't1' else 150,
        }
        model_params.update(params)
        
        # Train model
        model = CatBoostRegressor(**model_params)
        model.fit(train_pool, eval_set=val_pool)
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_pearson = evaluate_predictions(y_val, val_pred)
        
        return val_pearson
    
    return objective


def train_final_model(X_train, y_train, X_val, y_val, 
                      feature_names, best_params, target_name, output_dir):
    """Train final model with best params."""
    print(f"\n{'='*50}")
    print(f"Training final {target_name} model")
    print(f"{'='*50}")
    print(f"Best params: {best_params}")
    
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    
    model_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': CFG.SEED,
        'verbose': CFG.VERBOSE,
        'allow_writing_files': False,
        'task_type': 'CPU',
        'early_stopping_rounds': CFG.EARLY_STOPPING_ROUNDS,
    }
    model_params.update(best_params)
    
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool, eval_set=val_pool)
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_pearson = evaluate_predictions(y_val, val_pred)
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
    
    print(f"\nTop 15 features for {target_name}:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv(
        os.path.join(output_dir, f'feature_importance_{target_name}.csv'), 
        index=False
    )
    
    return val_pearson, importance_df


def main():
    print("="*60)
    print("Exp 16: CatBoost with Temporal Features + Optuna")
    print("="*60)
    
    # Output directory
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train_t0, y_train_t1, feature_cols = load_and_prepare_data(CFG.TRAIN_PATH)
    X_val, y_val_t0, y_val_t1, _ = load_and_prepare_data(CFG.VALID_PATH)
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Valid samples: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize W&B
    if WANDB_AVAILABLE:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="WunderNN-Predictorium",
            name="exp_16_optuna",
            config={
                "experiment": "exp_16",
                "model": "CatBoost",
                "n_trials": CFG.N_TRIALS,
                "features": len(feature_cols),
                "temporal_features": True
            }
        )
    
    results = {}
    best_params = {}
    
    # ========================================
    # Optimize t0
    # ========================================
    print("\n" + "="*60)
    print("Optimizing t0 hyperparameters")
    print("="*60)
    
    study_t0 = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CFG.SEED),
        study_name='exp_16_t0'
    )
    
    objective_t0 = create_objective(
        X_train, y_train_t0, X_val, y_val_t0, feature_cols, 't0'
    )
    
    study_t0.optimize(
        objective_t0,
        n_trials=CFG.N_TRIALS,
        timeout=CFG.TIMEOUT,
        show_progress_bar=True
    )
    
    print(f"\nBest t0 trial: {study_t0.best_trial.number}")
    print(f"Best t0 Pearson: {study_t0.best_value:.4f}")
    print(f"Best t0 params: {study_t0.best_params}")
    
    best_params['t0'] = study_t0.best_params
    
    # ========================================
    # Optimize t1 (with more regularization focus)
    # ========================================
    print("\n" + "="*60)
    print("Optimizing t1 hyperparameters (focus on regularization)")
    print("="*60)
    
    study_t1 = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CFG.SEED),
        study_name='exp_16_t1'
    )
    
    objective_t1 = create_objective(
        X_train, y_train_t1, X_val, y_val_t1, feature_cols, 't1'
    )
    
    study_t1.optimize(
        objective_t1,
        n_trials=CFG.N_TRIALS,
        timeout=CFG.TIMEOUT,
        show_progress_bar=True
    )
    
    print(f"\nBest t1 trial: {study_t1.best_trial.number}")
    print(f"Best t1 Pearson: {study_t1.best_value:.4f}")
    print(f"Best t1 params: {study_t1.best_params}")
    
    best_params['t1'] = study_t1.best_params
    
    # ========================================
    # Train final models with best params
    # ========================================
    print("\n" + "="*60)
    print("Training final models with best hyperparameters")
    print("="*60)
    
    pearson_t0, importance_t0 = train_final_model(
        X_train, y_train_t0, X_val, y_val_t0,
        feature_cols, best_params['t0'], 't0', output_dir
    )
    
    pearson_t1, importance_t1 = train_final_model(
        X_train, y_train_t1, X_val, y_val_t1,
        feature_cols, best_params['t1'], 't1', output_dir
    )
    
    results['t0'] = pearson_t0
    results['t1'] = pearson_t1
    
    # Save Optuna results
    study_t0.trials_dataframe().to_csv(
        os.path.join(output_dir, 'optuna_results_t0.csv'), index=False
    )
    study_t1.trials_dataframe().to_csv(
        os.path.join(output_dir, 'optuna_results_t1.csv'), index=False
    )
    
    # Save best params
    import json
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Log to W&B
    if WANDB_AVAILABLE:
        wandb.log({
            "val_pearson_t0": pearson_t0,
            "val_pearson_t1": pearson_t1,
            "val_pearson": (pearson_t0 + pearson_t1) / 2,
            "best_t0_trial": study_t0.best_trial.number,
            "best_t1_trial": study_t1.best_trial.number,
        })
        wandb.finish()
    
    # Final summary
    val_pearson = (pearson_t0 + pearson_t1) / 2
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Val Pearson t0: {pearson_t0:.4f}")
    print(f"Val Pearson t1: {pearson_t1:.4f}")
    print(f"Val Pearson (avg): {val_pearson:.4f}")
    print(f"\nModels saved to: {output_dir}")
    
    return val_pearson


if __name__ == "__main__":
    main()
