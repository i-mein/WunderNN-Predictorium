import numpy as np
import pandas as pd

def generate_features(df_seq):
    """
    Enhanced leakage-free feature generation for exp_07.
    
    Builds on exp_06's 65 features with additional cross-feature interactions
    and advanced rolling statistics.
    
    Total: 75 features (65 from exp_06 + 10 new)
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
    # 2. CROSS-FEATURE INTERACTIONS (NEW - 5 features)
    # ========================================
    
    # Price-volume interaction
    df['price_vol_interact'] = df['mid_price_0'] * df['vol_imbal_total']
    
    # Depth ratio (price difference normalized by spread)
    df['depth_ratio'] = (df['p0'] - df['p1']) / (df['spread_0'] + epsilon)
    
    # Level interaction (volume imbalance correlation)
    df['vol_imbal_level_interact'] = df['vol_imbal_0'] * df['vol_imbal_1']
    
    # Depth-volume interaction
    df['depth_vol_interact'] = df['price_depth'] * df['vol_ratio']
    
    # Spread momentum (will be computed after diffs)
    # Placeholder for now, will be filled after diff computation
    
    # ========================================
    # 3. ROLLING FEATURES (Conservative windows with min_periods=1)
    # ========================================
    
    roll_targets = ['mid_price_0', 'spread_0', 'vol_imbal_total']
    windows = [3, 10]  # Short-term only
    
    if 'seq_ix' in df.columns:
        # Grouped rolling to prevent cross-sequence leakage
        for target in roll_targets:
            grouped = df.groupby('seq_ix')[target]
            for w in windows:
                # min_periods=1 is CRITICAL - allows computation from step 0
                roll_mean = grouped.rolling(w, min_periods=1).mean()
                roll_std = grouped.rolling(w, min_periods=1).std().fillna(0)
                
                # Reset index to align with original dataframe
                df[f'{target}_roll_mean_{w}'] = roll_mean.reset_index(0, drop=True)
                df[f'{target}_roll_std_{w}'] = roll_std.reset_index(0, drop=True)
                
                # Advanced rolling stats for window=10 only (NEW - 5 features)
                if w == 10:
                    roll_min = grouped.rolling(w, min_periods=1).min()
                    roll_max = grouped.rolling(w, min_periods=1).max()
                    
                    df[f'{target}_roll_min_{w}'] = roll_min.reset_index(0, drop=True)
                    df[f'{target}_roll_max_{w}'] = roll_max.reset_index(0, drop=True)
                    df[f'{target}_roll_range_{w}'] = (
                        df[f'{target}_roll_max_{w}'] - df[f'{target}_roll_min_{w}']
                    )
    else:
        # Single sequence (inference mode)
        for target in roll_targets:
            for w in windows:
                df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
                df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std().fillna(0)
                
                if w == 10:
                    df[f'{target}_roll_min_{w}'] = df[target].rolling(w, min_periods=1).min()
                    df[f'{target}_roll_max_{w}'] = df[target].rolling(w, min_periods=1).max()
                    df[f'{target}_roll_range_{w}'] = (
                        df[f'{target}_roll_max_{w}'] - df[f'{target}_roll_min_{w}']
                    )
    
    # ========================================
    # 4. DIFFS (First-order differences with groupby)
    # ========================================
    
    diff_cols = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total']
    
    if 'seq_ix' in df.columns:
        # Grouped diff to prevent cross-sequence leakage
        grouped = df.groupby('seq_ix')[diff_cols]
        df_diff = grouped.diff().fillna(0)
        df_diff.columns = [f'{c}_diff' for c in diff_cols]
        df = pd.concat([df, df_diff], axis=1)
    else:
        # Single sequence (inference mode)
        for c in diff_cols:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
    
    return df

def get_feature_columns(df):
    """
    Returns list of feature columns, excluding metadata and targets.
    """
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def get_raw_feature_columns():
    """
    Returns list of raw feature column names (32 features).
    These are the original LOB features.
    """
    raw_cols = []
    # Prices: p0-p11
    raw_cols.extend([f'p{i}' for i in range(12)])
    # Volumes: v0-v11
    raw_cols.extend([f'v{i}' for i in range(12)])
    # Delta prices: dp0-dp3
    raw_cols.extend([f'dp{i}' for i in range(4)])
    # Delta volumes: dv0-dv3
    raw_cols.extend([f'dv{i}' for i in range(4)])
    return raw_cols

def get_engineered_feature_columns(df):
    """
    Returns list of engineered feature column names.
    These are all features except the raw 32.
    """
    all_features = get_feature_columns(df)
    raw_features = get_raw_feature_columns()
    return [f for f in all_features if f not in raw_features]
