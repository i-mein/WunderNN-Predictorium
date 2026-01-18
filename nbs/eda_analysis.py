import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
train_path = "datasets/train.parquet"
valid_path = "datasets/valid.parquet"

print(f"Loading {train_path}...")
df_train = pd.read_parquet(train_path)
print(f"Train shape: {df_train.shape}")
print(f"Columns: {df_train.columns.tolist()}")

print("\nMissing values:")
print(df_train.isnull().sum().sum())

print("\nTarget distribution:")
print(df_train[['t0', 't1']].describe())

# Check sequence lengths
seq_counts = df_train['seq_ix'].value_counts()
print("\nSequence length stats:")
print(seq_counts.describe())

# Check need_prediction count
print("\nNeed prediction counts:")
print(df_train['need_prediction'].value_counts())

# Correlation analysis
print("\nFeature-Target Correlation (Pearson):")
features = [c for c in df_train.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
corrs_t0 = df_train[features].corrwith(df_train['t0'])
corrs_t1 = df_train[features].corrwith(df_train['t1'])

print("\nTop 5 correlations with t0:")
print(corrs_t0.abs().sort_values(ascending=False).head(5))
print("\nTop 5 correlations with t1:")
print(corrs_t1.abs().sort_values(ascending=False).head(5))

# Check autocorrelation of targets within sequences
print("\nTarget Autocorrelation (lag 1):")
# We need to compute this per sequence to be accurate, but global lag 1 on sorted data (assuming sorted) is a rough proxy if we ignore boundaries.
# Better: groupby seq_ix
df_train['t0_lag1'] = df_train.groupby('seq_ix')['t0'].shift(1)
df_train['t1_lag1'] = df_train.groupby('seq_ix')['t1'].shift(1)

print("t0 auto-corr:", df_train['t0'].corr(df_train['t0_lag1']))
print("t1 auto-corr:", df_train['t1'].corr(df_train['t1_lag1']))

# Inspect first sequence
print("\nFirst sequence head:")
print(df_train[df_train['seq_ix'] == 0].head(10))
