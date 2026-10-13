"""
Feature generation for exp_14.

Uses the same leakage-free features as exp_06 (65 features).
This ensures fair comparison between DL and GBDT approaches.
"""
import numpy as np
import pandas as pd


def generate_features(df_seq):
    """
    Leakage-free feature generation.
    
    Identical to exp_06 features for fair comparison.
    65 features total:
    - 32 raw features (p0-p11, v0-v11, t0, t1, etc.)
    - 16 base engineered features
    - 12 rolling features
    - 5 diff features
    """
    df = df_seq.copy()
    
    # CRITICAL: Ensure proper ordering to prevent leakage
    if 'seq_ix' in df.columns and 'step_in_seq' in df.columns:
        df = df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
    
    epsilon = 1e-8
    
    # ========================================
    # 1. BASE FEATURES (Instantaneous, no leakage)
    # ========================================
    
    # Mid prices (best bid/ask average)
    df['mid_price_0'] = (df['p0'] + df['p6']) * 0.5
    df['mid_price_1'] = (df['p1'] + df['p7']) * 0.5
    
    # Bid-ask spreads
    df['spread_0'] = df['p6'] - df['p0']
    df['spread_1'] = df['p7'] - df['p1']
    
    # Volume imbalances per level
    for i in range(6):
        bv = df[f'v{i}']      # Bid volume at level i
        av = df[f'v{i+6}']    # Ask volume at level i
        df[f'vol_imbal_{i}'] = (bv - av) / (bv + av + epsilon)
    
    # Total volumes
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / \
                             (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)
    
    # Relative spread (normalized by price)
    df['relative_spread_0'] = df['spread_0'] / (df['mid_price_0'] + epsilon)
    
    # Volume ratio
    df['vol_ratio'] = df['total_bid_vol'] / (df['total_ask_vol'] + epsilon)
    
    # Price depth (bid side)
    df['price_depth'] = (df['p5'] - df['p0']) / (df['p0'] + epsilon)
    
    # ========================================
    # 2. ROLLING FEATURES (Conservative windows with min_periods=1)
    # ========================================
    
    roll_targets = ['mid_price_0', 'spread_0', 'vol_imbal_total']
    windows = [3, 10]
    
    if 'seq_ix' in df.columns:
        # Grouped rolling to prevent cross-sequence leakage
        for target in roll_targets:
            grouped = df.groupby('seq_ix')[target]
            for w in windows:
                roll_mean = grouped.rolling(w, min_periods=1).mean()
                roll_std = grouped.rolling(w, min_periods=1).std().fillna(0)
                
                df[f'{target}_roll_mean_{w}'] = roll_mean.reset_index(0, drop=True)
                df[f'{target}_roll_std_{w}'] = roll_std.reset_index(0, drop=True)
    else:
        # Single sequence (inference mode)
        for target in roll_targets:
            for w in windows:
                df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
                df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std().fillna(0)
    
    # ========================================
    # 3. DIFFS (First-order differences with groupby)
    # ========================================
    
    diff_cols = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total']
    
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')[diff_cols]
        df_diff = grouped.diff().fillna(0)
        df_diff.columns = [f'{c}_diff' for c in diff_cols]
        df = pd.concat([df, df_diff], axis=1)
    else:
        for c in diff_cols:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
    
    return df


def get_feature_columns(df):
    """
    Get feature column names (excluding targets and identifiers).
    """
    exclude = ['t0', 't1', 'seq_ix', 'step_in_seq', 'index']
    return [c for c in df.columns if c not in exclude and not c.startswith('_')]


def get_raw_feature_columns():
    """
    Get raw feature column names (prices and volumes only).
    """
    return [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)]
