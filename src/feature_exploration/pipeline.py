import sys
import os
import pandas as pd
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_exploration.generator import generate_candidate_features, get_feature_cols
from src.feature_exploration.selector import select_features, plot_importance

def main():
    warnings.simplefilter('ignore')
    
    # Load Data
    train_path = "datasets/train.parquet"
    print(f"Loading {train_path}...")
    df = pd.read_parquet(train_path)
    
    # Optimization: Sample sequences for Exploration to save time?
    # Full dataset is 1.4M rows. Generating hundreds of features might take time but is feasible.
    # Let's use 20% of sequences for speed.
    seq_ids = df['seq_ix'].unique()
    sample_seqs = seq_ids[:int(len(seq_ids)*0.2)]
    print(f"Subsampling {len(sample_seqs)} sequences for exploration...")
    df_sample = df[df['seq_ix'].isin(sample_seqs)].copy()
    
    # Generate
    print("Generating Features...")
    df_features = generate_candidate_features(df_sample)
    
    # Get cols
    all_cols = get_feature_cols(df_features)
    
    # Select
    print("Selecting Features...")
    # Target t0
    top_feats_t0, importances_t0 = select_features(df_features, all_cols, target_col='t0', top_k=50)
    
    # Plot
    plot_importance(importances_t0)
    
    # Save Report
    with open("feature_exploration_report.txt", "w") as f:
        f.write("Top 50 Features (t0):\n")
        f.write("\n".join(top_feats_t0))
        
    print("Pipeline Complete. See feature_exploration_report.txt")

if __name__ == "__main__":
    main()
