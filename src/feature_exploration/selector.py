import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns

def select_features(df, feature_cols, target_col='t0', correlation_threshold=0.95, top_k=50):
    """
    Selects top features by:
    1. Correlation Filter (Threshold 0.95 relaxed).
    2. CatBoost Feature Importance.
    """
    print(f"Starting selection with {len(feature_cols)} features...")
    
    # 1. Correlation Filter
    print("Calculating correlation matrix...")
    if len(df) > 100000:
        sample_df = df.sample(100000, random_state=42)
    else:
        sample_df = df
        
    corr_matrix = sample_df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    selected_cols = [c for c in feature_cols if c not in to_drop]
    print(f"Dropped {len(to_drop)} features due to correlation > {correlation_threshold}.")
    print(f"Remaining: {len(selected_cols)}")
    
    # 2. CatBoost Importance
    print("Training CatBoost for Importance Rank...")
    X = df[selected_cols]
    y = df[target_col]
    
    # Simple Train/Test split
    # CatBoost is robust.
    # We use CPU or GPU? Config decides. Default CPU is fine for analysis.
    
    train_pool = Pool(data=X, label=y)
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100,
        allow_writing_files=False
    )
    
    model.fit(train_pool, verbose=False)
    
    # Get importance
    feature_importances = model.get_feature_importance(train_pool)
    
    importances = pd.DataFrame({
        'feature': selected_cols,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Features:")
    print(importances.head(10))
    
    top_features = importances.head(top_k)['feature'].tolist()
    
    return top_features, importances

def plot_importance(importances, top_k=20):
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importances.head(top_k), x='importance', y='feature')
    plt.title(f"Top {top_k} Features (CatBoost)")
    plt.tight_layout()
    plt.savefig("feature_importance_catboost.png")
    print("Saved feature_importance_catboost.png")
