"""
Optuna hyperparameter optimization for exp_14: CatBoost.

Runs 3 seeds per trial and averages results for robust validation.
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


# ============================================================
# Configuration
# ============================================================
N_SEEDS = 1 #3
SEEDS = [42] #[42, 123, 456]
N_TRIALS = 50
TIMEOUT = 3600 * 2  # 2 hours max
EARLY_STOPPING = 100  # Faster during optimization


# ============================================================
# Data Loading
# ============================================================
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


# ============================================================
# Training with specific hyperparameters
# ============================================================
def train_single_seed(X_train, y_train, X_val, y_val, 
                      feature_names, params, seed, verbose=False):
    """Train a CatBoost model with given params and seed."""
    
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    
    model = CatBoostRegressor(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_strength=params['random_strength'],
        bagging_temperature=params['bagging_temperature'],
        min_data_in_leaf=params['min_data_in_leaf'],
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=seed,
        verbose=100 if verbose else 0,
        allow_writing_files=False,
        task_type='CPU',
        early_stopping_rounds=EARLY_STOPPING
    )
    
    model.fit(train_pool, eval_set=val_pool)
    
    val_pred = model.predict(X_val)
    val_pearson = evaluate_predictions(y_val, val_pred)
    
    return val_pearson, model


def train_multi_seed(X_train, y_train, X_val, y_val, 
                     feature_names, params, target_name, verbose=False):
    """Train with multiple seeds and return average Pearson."""
    
    pearson_scores = []
    
    for seed in SEEDS:
        pearson, _ = train_single_seed(
            X_train, y_train, X_val, y_val,
            feature_names, params, seed, verbose=verbose
        )
        pearson_scores.append(pearson)
    
    avg_pearson = np.mean(pearson_scores)
    std_pearson = np.std(pearson_scores)
    
    if verbose:
        print(f"\n{target_name} Results across {N_SEEDS} seeds:")
        print(f"  Scores: {[f'{p:.4f}' for p in pearson_scores]}")
        print(f"  Mean: {avg_pearson:.4f} ± {std_pearson:.4f}")
    
    return avg_pearson, std_pearson, pearson_scores


# ============================================================
# Optuna Objective
# ============================================================
class OptunaObjective:
    """Optuna objective for CatBoost hyperparameter optimization."""
    
    def __init__(self, X_train, y_train_t0, y_train_t1,
                 X_val, y_val_t0, y_val_t1, feature_names):
        self.X_train = X_train
        self.y_train_t0 = y_train_t0
        self.y_train_t1 = y_train_t1
        self.X_val = X_val
        self.y_val_t0 = y_val_t0
        self.y_val_t1 = y_val_t1
        self.feature_names = feature_names
        
    def __call__(self, trial):
        # Sample hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 5000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        }
        
        # Train t0 model with multi-seed
        avg_pearson_t0, std_t0, _ = train_multi_seed(
            self.X_train, self.y_train_t0, self.X_val, self.y_val_t0,
            self.feature_names, params, 't0', verbose=False
        )
        
        # Train t1 model with multi-seed
        avg_pearson_t1, std_t1, _ = train_multi_seed(
            self.X_train, self.y_train_t1, self.X_val, self.y_val_t1,
            self.feature_names, params, 't1', verbose=False
        )
        
        # Combined score (average of t0 and t1)
        combined_score = (avg_pearson_t0 + avg_pearson_t1) / 2
        
        # Log intermediate values
        trial.set_user_attr('pearson_t0', avg_pearson_t0)
        trial.set_user_attr('pearson_t1', avg_pearson_t1)
        trial.set_user_attr('std_t0', std_t0)
        trial.set_user_attr('std_t1', std_t1)
        
        print(f"\nTrial {trial.number}: Combined={combined_score:.4f} "
              f"(t0={avg_pearson_t0:.4f}±{std_t0:.4f}, "
              f"t1={avg_pearson_t1:.4f}±{std_t1:.4f})")
        
        return combined_score


# ============================================================
# Final Training with Best Params
# ============================================================
def train_final_models(X_train, y_train_t0, y_train_t1,
                       X_val, y_val_t0, y_val_t1,
                       feature_names, best_params, output_dir):
    """Train final models with best params and save."""
    
    print("\n" + "="*60)
    print("Training Final Models with Best Hyperparameters")
    print("="*60)
    print(f"\nBest params: {best_params}")
    
    # Use more iterations for final training
    final_params = best_params.copy()
    final_params['iterations'] = min(best_params['iterations'] * 2, 10000)
    
    results = {}
    
    for target_name, y_train, y_val in [('t0', y_train_t0, y_val_t0), 
                                         ('t1', y_train_t1, y_val_t1)]:
        print(f"\n{'='*40}")
        print(f"Training final {target_name} model")
        print(f"{'='*40}")
        
        # Train with multiple seeds
        avg_pearson, std_pearson, scores = train_multi_seed(
            X_train, y_train, X_val, y_val,
            feature_names, final_params, target_name, verbose=True
        )
        
        results[target_name] = {
            'avg_pearson': avg_pearson,
            'std_pearson': std_pearson,
            'scores': scores
        }
        
        # Train final model with seed 42 for submission
        print(f"\nTraining final {target_name} model (seed=42) for submission...")
        train_pool = Pool(X_train, y_train, feature_names=feature_names)
        val_pool = Pool(X_val, y_val, feature_names=feature_names)
        
        final_model = CatBoostRegressor(
            iterations=final_params['iterations'],
            learning_rate=final_params['learning_rate'],
            depth=final_params['depth'],
            l2_leaf_reg=final_params['l2_leaf_reg'],
            random_strength=final_params['random_strength'],
            bagging_temperature=final_params['bagging_temperature'],
            min_data_in_leaf=final_params['min_data_in_leaf'],
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=100,
            allow_writing_files=False,
            task_type='CPU',
            early_stopping_rounds=200
        )
        
        final_model.fit(train_pool, eval_set=val_pool)
        
        # Save model
        model_name = CFG.MODEL_T0 if target_name == 't0' else CFG.MODEL_T1
        model_path = os.path.join(output_dir, model_name)
        final_model.save_model(model_path)
        print(f"Saved {target_name} model to {model_path}")
        
        # Feature importance
        importance = final_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 features for {target_name}:")
        print(importance_df.head(10).to_string(index=False))
    
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Exp 14: CatBoost Hyperparameter Optimization with Optuna")
    print(f"Using {N_SEEDS} seeds per trial for robust validation")
    print("="*60)
    
    # Output directory
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data
    X_train, y_train_t0, y_train_t1, feature_cols = load_and_prepare_data(
        CFG.TRAIN_PATH
    )
    X_val, y_val_t0, y_val_t1, _ = load_and_prepare_data(
        CFG.VALID_PATH
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Valid samples: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize W&B
    if WANDB_AVAILABLE:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="WunderNN-Predictorium",
            name="exp_14_optuna",
            config={
                "experiment": "exp_14",
                "model": "CatBoost",
                "n_seeds": N_SEEDS,
                "n_trials": N_TRIALS,
                "features": len(feature_cols)
            }
        )
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='exp_14_catboost'
    )
    
    # Create objective
    objective = OptunaObjective(
        X_train, y_train_t0, y_train_t1,
        X_val, y_val_t0, y_val_t1,
        feature_cols
    )
    
    # Optimize
    print(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    
    # Print best results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best combined Pearson: {study.best_value:.4f}")
    print(f"  t0: {study.best_trial.user_attrs['pearson_t0']:.4f} "
          f"± {study.best_trial.user_attrs['std_t0']:.4f}")
    print(f"  t1: {study.best_trial.user_attrs['pearson_t1']:.4f} "
          f"± {study.best_trial.user_attrs['std_t1']:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study results
    results_df = study.trials_dataframe()
    results_path = os.path.join(output_dir, 'optuna_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved optimization results to {results_path}")
    
    # Train final models with best params
    final_results = train_final_models(
        X_train, y_train_t0, y_train_t1,
        X_val, y_val_t0, y_val_t1,
        feature_cols, study.best_params, output_dir
    )
    
    # Log to W&B
    if WANDB_AVAILABLE:
        wandb.log({
            "best_combined_pearson": study.best_value,
            "best_pearson_t0": study.best_trial.user_attrs['pearson_t0'],
            "best_pearson_t1": study.best_trial.user_attrs['pearson_t1'],
            "final_pearson_t0": final_results['t0']['avg_pearson'],
            "final_pearson_t1": final_results['t1']['avg_pearson'],
            **{f"best_{k}": v for k, v in study.best_params.items()}
        })
        wandb.finish()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    combined = (final_results['t0']['avg_pearson'] + 
                final_results['t1']['avg_pearson']) / 2
    print(f"\nFinal Combined Pearson: {combined:.4f}")
    print(f"  t0: {final_results['t0']['avg_pearson']:.4f} "
          f"± {final_results['t0']['std_pearson']:.4f}")
    print(f"  t1: {final_results['t1']['avg_pearson']:.4f} "
          f"± {final_results['t1']['std_pearson']:.4f}")
    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
