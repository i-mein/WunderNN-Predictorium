import numpy as np
import pandas as pd

def generate_features(df_seq):
    """
    Generates engineered features including rolling stats.
    """
    # Defensive copy
    df = df_seq.copy()
    
    # --- PHASE 1: Step-wise Engineering (Same as Exp 02) ---
    
    # 1. Mid Prices
    df['mid_price_0'] = (df['p0'] + df['p6']) / 2
    df['mid_price_1'] = (df['p1'] + df['p7']) / 2
    
    # 2. Bid-Ask Spreads
    df['spread_0'] = df['p6'] - df['p0']
    df['spread_1'] = df['p7'] - df['p1']
    
    # 3. Volume Imbalances
    epsilon = 1e-8
    for i in range(6):
        bv = df[f'v{i}']
        av = df[f'v{i+6}']
        df[f'vol_imbal_{i}'] = (bv - av) / (bv + av + epsilon)
        
    # 4. Accumulated Volumes
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)

    # --- PHASE 2: Rolling Statistics ---
    # Rolling on Select Features
    roll_feats = ['mid_price_0', 'spread_0', 'vol_imbal_total']
    windows = [10, 50]
    
    # We must group by sequence if multiple sequences exist
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')[roll_feats]
        for w in windows:
            # Shift by 1? No, rolling includes current. Standard.
            # But for prediction at t, can we use t? Yes, we have market state at t via Limit Order Book.
            # LOB snapshot is instantaneous. Rolling mean of last 10 snapshots (inclusive) is valid feature.
            # Note: Pandas rolling is inclusive of current row by default.
            
            # Mean
            roll_mean = grouped.rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
            roll_mean.columns = [f'{c}_roll_mean_{w}' for c in roll_feats]
            df = pd.concat([df, roll_mean], axis=1)
            
            # Std
            roll_std = grouped.rolling(window=w, min_periods=1).std().fillna(0).reset_index(0, drop=True)
            roll_std.columns = [f'{c}_roll_std_{w}' for c in roll_feats]
            df = pd.concat([df, roll_std], axis=1)
    else:
        # Single sequence case
        for w in windows:
            for c in roll_feats:
                df[f'{c}_roll_mean_{w}'] = df[c].rolling(window=w, min_periods=1).mean()
                df[f'{c}_roll_std_{w}'] = df[c].rolling(window=w, min_periods=1).std().fillna(0)

    # --- PHASE 3: Differencing (Same as Exp 02) ---
    cols_to_diff = ['p0', 'p6', 'mid_price_0', 'spread_0']
    
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')[cols_to_diff]
        df_diff = grouped.diff().fillna(0)
        df_diff.columns = [f'{c}_diff' for c in cols_to_diff]
        df = pd.concat([df, df_diff], axis=1)
    else:
        for c in cols_to_diff:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
            
    return df

def get_feature_columns(df):
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
    return [c for c in df.columns if c not in exclude]
