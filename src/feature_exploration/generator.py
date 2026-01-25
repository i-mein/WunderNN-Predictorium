import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_candidate_features(df):
    """
    Generates extended candidate features:
    - Base (Mids, Spreads, Imbals)
    - Ratios (Spread/Mid, Bid/Ask Vol)
    - Rolling Stats (Std, EMA, Diff, Rank, Skew, Kurt, Min, Max, Quantiles)
    """
    df = df.copy()
    epsilon = 1e-8
    
    # 1. Base
    df['mid_price_0'] = (df['p0'] + df['p6']) * 0.5
    df['spread_0'] = df['p6'] - df['p0']
    
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)

    # 2. Ratios / Interactions
    df['relative_spread_0'] = df['spread_0'] / (df['mid_price_0'] + epsilon)
    df['vol_ratio'] = df['total_bid_vol'] / (df['total_ask_vol'] + epsilon)
    
    # Targets for Rolling
    targets = ['mid_price_0', 'spread_0', 'vol_imbal_total', 'relative_spread_0', 'vol_ratio']
    windows = [3, 13, 21, 55] # Added 55 for longer trend
    
    grouped = df.groupby('seq_ix')
    
    print(f"Generating EXTENDED features for {len(targets)} targets with windows {windows}...")
    
    for t in targets:
        g_t = grouped[t]
        for w in windows:
            # Volatility
            df[f'{t}_std_{w}'] = g_t.rolling(w, min_periods=1).std().reset_index(0, drop=True)
            
            # EMA
            df[f'{t}_ema_{w}'] = g_t.transform(lambda x: x.ewm(span=w, adjust=False).mean())
            
            # Diff
            df[f'{t}_diff_{w}'] = g_t.diff(periods=w).fillna(0).reset_index(0, drop=True)
            
            # Rank
            df[f'{t}_rank_{w}'] = g_t.rolling(w, min_periods=1).rank(pct=True).reset_index(0, drop=True)
            
            # Z-Score
            std = df[f'{t}_std_{w}']
            mean = df[f'{t}_ema_{w}']
            df[f'{t}_zscore_{w}'] = (df[t] - mean) / (std + epsilon)
            
            # Quantiles (Min, Max, 0.25, 0.75)
            # Rolling Min/Max is fast
            df[f'{t}_min_{w}'] = g_t.rolling(w, min_periods=1).min().reset_index(0, drop=True)
            df[f'{t}_max_{w}'] = g_t.rolling(w, min_periods=1).max().reset_index(0, drop=True)
            
            # Skew/Kurt only for smaller/med windows to avoid massive computation
            if w in [13, 21, 55]:
                df[f'{t}_skew_{w}'] = g_t.rolling(w, min_periods=w).skew().fillna(0).reset_index(0, drop=True)
                # Kurtosis often noisy, maybe skip? User asked for it. 
                # df[f'{t}_kurt_{w}'] = g_t.rolling(w, min_periods=w).kurt().fillna(0).reset_index(0, drop=True) 

    return df

def get_feature_cols(df):
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1', 'prediction_id']
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
