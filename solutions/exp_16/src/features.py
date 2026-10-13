"""
Enhanced feature generation for exp_16.

Adds temporal/momentum features optimized for t1 (longer horizon prediction).
Focus on:
- Longer rolling windows (20, 50 steps)
- Momentum/trend indicators
- Acceleration features (second-order diffs)
- Cumulative features
"""
import numpy as np
import pandas as pd


def generate_features(df_seq):
    """
    Feature generation with temporal/momentum features for t1.
    
    Features:
    - Base features from exp_15 (~65)
    - Micro-price features (~10)
    - Order book pressure (~12)
    - NEW: Temporal/momentum features (~25)
    
    Total: ~110+ features
    """
    df = df_seq.copy()
    
    # CRITICAL: Ensure proper ordering to prevent leakage
    if 'seq_ix' in df.columns and 'step_in_seq' in df.columns:
        df = df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
    
    epsilon = 1e-8
    
    # ========================================
    # 1. BASE FEATURES (Instantaneous)
    # ========================================
    
    # Mid prices
    df['mid_price_0'] = (df['p0'] + df['p6']) * 0.5
    df['mid_price_1'] = (df['p1'] + df['p7']) * 0.5
    
    # Spreads
    df['spread_0'] = df['p6'] - df['p0']
    df['spread_1'] = df['p7'] - df['p1']
    
    # Volume imbalances per level
    for i in range(6):
        bv = df[f'v{i}']
        av = df[f'v{i+6}']
        df[f'vol_imbal_{i}'] = (bv - av) / (bv + av + epsilon)
    
    # Total volumes
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / \
                             (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)
    
    # Relative spread
    df['relative_spread_0'] = df['spread_0'] / (df['mid_price_0'] + epsilon)
    
    # Volume ratio
    df['vol_ratio'] = df['total_bid_vol'] / (df['total_ask_vol'] + epsilon)
    
    # Price depth
    df['price_depth'] = (df['p5'] - df['p0']) / (df['p0'] + epsilon)
    
    # ========================================
    # 2. MICRO-PRICE FEATURES
    # ========================================
    
    df['micro_price'] = (df['p0'] * df['v6'] + df['p6'] * df['v0']) / \
                        (df['v0'] + df['v6'] + epsilon)
    df['micro_price_deviation'] = df['micro_price'] - df['mid_price_0']
    df['micro_price_deviation_norm'] = df['micro_price_deviation'] / (df['spread_0'] + epsilon)
    
    df['weighted_mid_2'] = (df['p1'] * df['v7'] + df['p7'] * df['v1']) / \
                           (df['v1'] + df['v7'] + epsilon)
    df['micro_price_deviation_2'] = df['weighted_mid_2'] - df['mid_price_1']
    
    # ========================================
    # 3. ORDER BOOK PRESSURE
    # ========================================
    
    df['bid_pressure_0'] = df['v0'] / (df['total_bid_vol'] + epsilon)
    df['bid_pressure_01'] = (df['v0'] + df['v1']) / (df['total_bid_vol'] + epsilon)
    df['ask_pressure_0'] = df['v6'] / (df['total_ask_vol'] + epsilon)
    df['ask_pressure_01'] = (df['v6'] + df['v7']) / (df['total_ask_vol'] + epsilon)
    
    # LOB slope
    bid_vwap = sum(df[f'p{i}'] * df[f'v{i}'] for i in range(6)) / (df['total_bid_vol'] + epsilon)
    ask_vwap = sum(df[f'p{i+6}'] * df[f'v{i+6}'] for i in range(6)) / (df['total_ask_vol'] + epsilon)
    
    df['bid_slope'] = (df['p0'] - bid_vwap) / (df['total_bid_vol'] + epsilon)
    df['ask_slope'] = (ask_vwap - df['p6']) / (df['total_ask_vol'] + epsilon)
    df['slope_imbal'] = (df['bid_slope'] - df['ask_slope']) / \
                        (df['bid_slope'] + df['ask_slope'] + epsilon)
    
    df['depth_imbal_near'] = (df['v0'] + df['v1'] - df['v6'] - df['v7']) / \
                             (df['v0'] + df['v1'] + df['v6'] + df['v7'] + epsilon)
    df['depth_imbal_far'] = (df['v4'] + df['v5'] - df['v10'] - df['v11']) / \
                            (df['v4'] + df['v5'] + df['v10'] + df['v11'] + epsilon)
    
    df['bid_vwap'] = bid_vwap
    df['ask_vwap'] = ask_vwap
    df['vwap_spread'] = ask_vwap - bid_vwap
    
    # Price level features
    df['price_range'] = df['p11'] - df['p0']
    df['bid_range'] = df['p5'] - df['p0']
    df['ask_range'] = df['p11'] - df['p6']
    df['bid_spacing'] = (df['p0'] - df['p5']) / 5
    df['ask_spacing'] = (df['p11'] - df['p6']) / 5
    df['spacing_ratio'] = df['bid_spacing'] / (df['ask_spacing'] + epsilon)
    
    # ========================================
    # 4. TEMPORAL/MOMENTUM FEATURES (NEW for t1)
    # ========================================
    
    # Longer rolling windows for t1 (which predicts further ahead)
    roll_targets_short = ['mid_price_0', 'spread_0', 'vol_imbal_total', 
                          'micro_price_deviation', 'depth_imbal_near']
    windows_short = [3, 10]
    
    # Additional longer windows for momentum
    roll_targets_long = ['mid_price_0', 'vol_imbal_total', 'micro_price_deviation']
    windows_long = [20, 50]
    
    if 'seq_ix' in df.columns:
        # Short windows (standard)
        for target in roll_targets_short:
            grouped = df.groupby('seq_ix')[target]
            for w in windows_short:
                roll_mean = grouped.rolling(w, min_periods=1).mean()
                roll_std = grouped.rolling(w, min_periods=1).std().fillna(0)
                df[f'{target}_roll_mean_{w}'] = roll_mean.reset_index(0, drop=True)
                df[f'{target}_roll_std_{w}'] = roll_std.reset_index(0, drop=True)
        
        # Long windows (momentum for t1)
        for target in roll_targets_long:
            grouped = df.groupby('seq_ix')[target]
            for w in windows_long:
                roll_mean = grouped.rolling(w, min_periods=1).mean()
                roll_std = grouped.rolling(w, min_periods=1).std().fillna(0)
                roll_min = grouped.rolling(w, min_periods=1).min()
                roll_max = grouped.rolling(w, min_periods=1).max()
                
                df[f'{target}_roll_mean_{w}'] = roll_mean.reset_index(0, drop=True)
                df[f'{target}_roll_std_{w}'] = roll_std.reset_index(0, drop=True)
                # Range (volatility proxy)
                df[f'{target}_roll_range_{w}'] = (roll_max - roll_min).reset_index(0, drop=True)
    else:
        # Single sequence (inference mode)
        for target in roll_targets_short:
            for w in windows_short:
                df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
                df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std().fillna(0)
        
        for target in roll_targets_long:
            for w in windows_long:
                df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
                df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std().fillna(0)
                roll_min = df[target].rolling(w, min_periods=1).min()
                roll_max = df[target].rolling(w, min_periods=1).max()
                df[f'{target}_roll_range_{w}'] = roll_max - roll_min
    
    # ========================================
    # 5. MOMENTUM INDICATORS (NEW)
    # ========================================
    
    # Price momentum (short-term trend)
    if 'seq_ix' in df.columns:
        grouped_mid = df.groupby('seq_ix')['mid_price_0']
        # Momentum: current - lagged value
        df['momentum_5'] = df['mid_price_0'] - grouped_mid.shift(5).bfill()
        df['momentum_10'] = df['mid_price_0'] - grouped_mid.shift(10).bfill()
        df['momentum_20'] = df['mid_price_0'] - grouped_mid.shift(20).bfill()
        
        # Volume momentum
        grouped_vol = df.groupby('seq_ix')['vol_imbal_total']
        df['vol_momentum_5'] = df['vol_imbal_total'] - grouped_vol.shift(5).bfill()
        df['vol_momentum_10'] = df['vol_imbal_total'] - grouped_vol.shift(10).bfill()
        
        # Micro-price momentum
        grouped_micro = df.groupby('seq_ix')['micro_price_deviation']
        df['micro_momentum_5'] = df['micro_price_deviation'] - grouped_micro.shift(5).bfill()
        df['micro_momentum_10'] = df['micro_price_deviation'] - grouped_micro.shift(10).bfill()
    else:
        df['momentum_5'] = df['mid_price_0'] - df['mid_price_0'].shift(5).bfill()
        df['momentum_10'] = df['mid_price_0'] - df['mid_price_0'].shift(10).bfill()
        df['momentum_20'] = df['mid_price_0'] - df['mid_price_0'].shift(20).bfill()
        df['vol_momentum_5'] = df['vol_imbal_total'] - df['vol_imbal_total'].shift(5).bfill()
        df['vol_momentum_10'] = df['vol_imbal_total'] - df['vol_imbal_total'].shift(10).bfill()
        df['micro_momentum_5'] = df['micro_price_deviation'] - df['micro_price_deviation'].shift(5).bfill()
        df['micro_momentum_10'] = df['micro_price_deviation'] - df['micro_price_deviation'].shift(10).bfill()
    
    # ========================================
    # 6. ACCELERATION FEATURES (2nd order diffs)
    # ========================================
    
    diff_cols = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total',
                 'micro_price', 'micro_price_deviation', 'depth_imbal_near']
    
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')[diff_cols]
        df_diff = grouped.diff().fillna(0)
        df_diff.columns = [f'{c}_diff' for c in diff_cols]
        df = pd.concat([df, df_diff], axis=1)
        
        # 2nd order diff (acceleration) for key features
        accel_cols = ['mid_price_0_diff', 'vol_imbal_total_diff', 'micro_price_deviation_diff']
        for col in accel_cols:
            grouped_accel = df.groupby('seq_ix')[col]
            df[f'{col}_accel'] = grouped_accel.diff().fillna(0)
    else:
        for c in diff_cols:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
        
        accel_cols = ['mid_price_0_diff', 'vol_imbal_total_diff', 'micro_price_deviation_diff']
        for col in accel_cols:
            df[f'{col}_accel'] = df[col].diff().fillna(0)
    
    # ========================================
    # 7. TREND STRENGTH INDICATORS
    # ========================================
    
    # Trend strength: ratio of momentum to volatility (like Sharpe)
    df['trend_strength_10'] = df['momentum_10'] / (df['mid_price_0_roll_std_10'] + epsilon)
    df['trend_strength_20'] = df['momentum_20'] / (df['mid_price_0_roll_std_20'] + epsilon)
    
    # Volume trend strength
    df['vol_trend_strength'] = df['vol_momentum_10'] / (df['vol_imbal_total_roll_std_10'] + epsilon)
    
    # Price relative to recent range (0-1 normalized)
    df['price_position_20'] = (df['mid_price_0'] - df['mid_price_0_roll_mean_20']) / \
                              (df['mid_price_0_roll_range_20'] + epsilon)
    df['price_position_50'] = (df['mid_price_0'] - df['mid_price_0_roll_mean_50']) / \
                              (df['mid_price_0_roll_range_50'] + epsilon)
    
    return df


def get_feature_columns(df):
    """Get list of feature columns (excludes targets and metadata)."""
    exclude_cols = {'t0', 't1', 'seq_ix', 'step_in_seq', 'index'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if not c.startswith('_')]
    return feature_cols
