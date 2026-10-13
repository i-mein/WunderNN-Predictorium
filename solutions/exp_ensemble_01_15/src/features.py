"""
Enhanced feature generation for exp_15.

Adds micro-price and trade flow features to exp_14's 65 features.
Total: ~85 features
"""
import numpy as np
import pandas as pd


def generate_features(df_seq):
    """
    Enhanced feature generation with micro-price and trade flow.
    
    Features:
    - 65 base features from exp_14 (raw, engineered, rolling, diffs)
    - ~10 micro-price features
    - ~10 trade flow / order book pressure features
    
    Total: ~85 features
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
    # 2. MICRO-PRICE FEATURES (NEW)
    # ========================================
    
    # Micro-price: Volume-weighted mid price
    # More accurate fair value estimate than simple mid
    df['micro_price'] = (df['p0'] * df['v6'] + df['p6'] * df['v0']) / \
                        (df['v0'] + df['v6'] + epsilon)
    
    # Micro-price deviation from mid (indicates directional pressure)
    df['micro_price_deviation'] = df['micro_price'] - df['mid_price_0']
    
    # Normalized micro-price deviation
    df['micro_price_deviation_norm'] = df['micro_price_deviation'] / \
                                       (df['spread_0'] + epsilon)
    
    # Weighted mid-price using level 2 volumes
    df['weighted_mid_2'] = (df['p1'] * df['v7'] + df['p7'] * df['v1']) / \
                           (df['v1'] + df['v7'] + epsilon)
    
    # Level 2 micro-price deviation
    df['micro_price_deviation_2'] = df['weighted_mid_2'] - df['mid_price_1']
    
    # ========================================
    # 3. ORDER BOOK PRESSURE / TRADE FLOW (NEW)
    # ========================================
    
    # Bid pressure (volume concentration at best bid)
    df['bid_pressure_0'] = df['v0'] / (df['total_bid_vol'] + epsilon)
    df['bid_pressure_01'] = (df['v0'] + df['v1']) / (df['total_bid_vol'] + epsilon)
    
    # Ask pressure (volume concentration at best ask)
    df['ask_pressure_0'] = df['v6'] / (df['total_ask_vol'] + epsilon)
    df['ask_pressure_01'] = (df['v6'] + df['v7']) / (df['total_ask_vol'] + epsilon)
    
    # LOB slope (price impact per unit volume)
    # Bid side slope
    bid_vwap = sum(df[f'p{i}'] * df[f'v{i}'] for i in range(6)) / \
               (df['total_bid_vol'] + epsilon)
    df['bid_slope'] = (df['p0'] - bid_vwap) / (df['total_bid_vol'] + epsilon)
    
    # Ask side slope
    ask_vwap = sum(df[f'p{i+6}'] * df[f'v{i+6}'] for i in range(6)) / \
               (df['total_ask_vol'] + epsilon)
    df['ask_slope'] = (ask_vwap - df['p6']) / (df['total_ask_vol'] + epsilon)
    
    # Slope imbalance
    df['slope_imbal'] = (df['bid_slope'] - df['ask_slope']) / \
                        (df['bid_slope'] + df['ask_slope'] + epsilon)
    
    # Order book depth imbalance (volume at different levels)
    df['depth_imbal_near'] = (df['v0'] + df['v1'] - df['v6'] - df['v7']) / \
                             (df['v0'] + df['v1'] + df['v6'] + df['v7'] + epsilon)
    df['depth_imbal_far'] = (df['v4'] + df['v5'] - df['v10'] - df['v11']) / \
                            (df['v4'] + df['v5'] + df['v10'] + df['v11'] + epsilon)
    
    # Volume-weighted price level
    df['bid_vwap'] = bid_vwap
    df['ask_vwap'] = ask_vwap
    df['vwap_spread'] = ask_vwap - bid_vwap
    
    # ========================================
    # 4. PRICE LEVEL FEATURES (NEW)
    # ========================================
    
    # Price range (volatility proxy)
    df['price_range'] = df['p11'] - df['p0']  # Full depth range
    df['bid_range'] = df['p5'] - df['p0']
    df['ask_range'] = df['p11'] - df['p6']
    
    # Price level spacing (liquidity indicator)
    df['bid_spacing'] = (df['p0'] - df['p5']) / 5  # Average bid tick spacing
    df['ask_spacing'] = (df['p11'] - df['p6']) / 5  # Average ask tick spacing
    df['spacing_ratio'] = df['bid_spacing'] / (df['ask_spacing'] + epsilon)
    
    # ========================================
    # 5. ROLLING FEATURES (Conservative windows)
    # ========================================
    
    roll_targets = ['mid_price_0', 'spread_0', 'vol_imbal_total', 
                    'micro_price_deviation', 'depth_imbal_near']
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
    # 6. DIFFS (First-order differences)
    # ========================================
    
    diff_cols = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total',
                 'micro_price', 'micro_price_deviation', 'depth_imbal_near']
    
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
    Get list of feature columns (excludes targets and metadata).
    """
    # Columns to exclude
    exclude_cols = {'t0', 't1', 'seq_ix', 'step_in_seq', 'index'}
    
    # Get all numeric columns that are not excluded
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Also filter out any potential non-feature columns
    feature_cols = [c for c in feature_cols if not c.startswith('_')]
    
    return feature_cols
