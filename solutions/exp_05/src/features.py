import numpy as np
import pandas as pd

def generate_features(df_seq):
    """
    Exp 05 Features: Extended Set from Exploration.
    Includes: Base, RSI/BB (Exp 04), Min/Max/Skew (55), Ratios.
    STRICT LEAKAGE PREVENTION: Sort by sequence and step.
    """
    # 0. Leakage Prevention
    # Ensure correct order
    df = df_seq.copy()
    if 'seq_ix' in df.columns and 'step_in_seq' in df.columns:
        df = df.sort_values(['seq_ix', 'step_in_seq'])
    
    # 1. Base (Exp 02)
    df['mid_price_0'] = (df['p0'] + df['p6']) * 0.5
    df['mid_price_1'] = (df['p1'] + df['p7']) * 0.5
    df['spread_0'] = df['p6'] - df['p0']
    df['spread_1'] = df['p7'] - df['p1']
    
    epsilon = 1e-8
    for i in range(6):
        bv = df[f'v{i}']
        av = df[f'v{i+6}']
        df[f'vol_imbal_{i}'] = (bv - av) / (bv + av + epsilon)
        
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)

    # 2. Ratios (New)
    df['relative_spread_0'] = df['spread_0'] / (df['mid_price_0'] + epsilon)
    df['vol_ratio'] = df['total_bid_vol'] / (df['total_ask_vol'] + epsilon)

    # 3. Rolling Features
    # Targets: mid0, spread_0, vol_imbal_total
    targets = ['mid_price_0', 'spread_0', 'vol_imbal_total']
    
    # Group for rolling
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')
    else:
        # If single sequence (inference)
        grouped = None
        
    for t in targets:
        # Helper to apply rolling
        if grouped:
            g_t = grouped[t]
        else:
            g_t = df[t] # Series object has rolling
        
        # Exp 04 Features (RSI 14, BB 20)
        # We reimplement or copy? Reimplement consistent with Exploration findings.
        # But RSI/BB (Stationary) were good in Exp 04 Val.
        # Exploration found Z-Score 3 good. BB %B is Z-Score on price.
        # Let's keep BB %B (20).
        
        # Windows
        # 3 (Short momentum/Z), 21 (Med), 55 (Long Support/Analysis)
        windows = [3, 21, 55]
        
        for w in windows:
            if grouped:
                roll = g_t.rolling(w, min_periods=1)
            else:
                roll = g_t.rolling(w, min_periods=1)
                
            mean = roll.mean().reset_index(0, drop=True) if grouped else roll.mean()
            std = roll.std().reset_index(0, drop=True) if grouped else roll.std()
            
            # Z-Score (Regime)
            # Replaces BB %B (similar info)
            df[f'{t}_zscore_{w}'] = (df[t] - mean) / (std.fillna(0) + epsilon)
            
            # Volatility
            df[f'{t}_vol_{w}'] = std.fillna(0)
            
            # Min/Max (Support/Resistance) - Especially 55
            df[f'{t}_min_{w}'] = roll.min().reset_index(0, drop=True) if grouped else roll.min()
            df[f'{t}_max_{w}'] = roll.max().reset_index(0, drop=True) if grouped else roll.max()
            
            # Rank? CatBoost liked it.
            # df[f'{t}_rank_{w}'] = roll.rank(pct=True)... (Rank is slow in Pandas?)
            # Skip rank for speed/memory unless vital. CatBoost liked min/max/zscore more.
            # Ratios had rank 3.
            
            # Skew (Only 55)
            if w == 55:
                if grouped:
                    roll_skew = g_t.rolling(w, min_periods=w).skew().reset_index(0, drop=True)
                else:
                    roll_skew = g_t.rolling(w, min_periods=w).skew()
                df[f'{t}_skew_{w}'] = roll_skew.fillna(0)
    
    # 4. Diffs (Standard)
    cols_to_diff = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total']
    if grouped:
        diffs = grouped[cols_to_diff].diff().fillna(0)
        # Rename cols? Just concat
        diffs.columns = [f'{c}_diff' for c in cols_to_diff]
        # Align index? groupby diff preserves index usually.
        # But verify.
        df = pd.concat([df, diffs], axis=1)
    else:
        for c in cols_to_diff:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
            
    return df

def get_feature_columns(df):
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
