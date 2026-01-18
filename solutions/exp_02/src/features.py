import numpy as np
import pandas as pd

def generate_features(df_seq):
    """
    Generates engineered features for a single sequence or batch of dataframe.
    Assumes df_seq contains columns p0..p11, v0..v11, etc.
    """
    # Defensive copy
    df = df_seq.copy()
    
    # 1. Mid Prices
    # p0 is best bid, p6 is best ask (based on typical LOB structure, let's verify)
    # README says:
    # p0..p5: Bid Price Features.
    # p6..p11: Ask Price Features.
    # Usually p0 is best bid (highest), p6 is best ask (lowest).
    
    # Mid Price 0 (Best Quote)
    df['mid_price_0'] = (df['p0'] + df['p6']) / 2
    
    # Mid Price 1 (Second level)
    df['mid_price_1'] = (df['p1'] + df['p7']) / 2
    
    # 2. Bid-Ask Spreads
    df['spread_0'] = df['p6'] - df['p0']
    df['spread_1'] = df['p7'] - df['p1']
    
    # 3. Volume Imbalances
    # (BidVol - AskVol) / (BidVol + AskVol)
    # v0..v5 (Bid), v6..v11 (Ask)
    epsilon = 1e-8
    for i in range(6):
        bv = df[f'v{i}']
        av = df[f'v{i+6}']
        df[f'vol_imbal_{i}'] = (bv - av) / (bv + av + epsilon)
        
    # 4. Accumulated Volumes (Total liquidity)
    df['total_bid_vol'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    df['total_ask_vol'] = df[[f'v{i+6}' for i in range(6)]].sum(axis=1)
    df['vol_imbal_total'] = (df['total_bid_vol'] - df['total_ask_vol']) / (df['total_bid_vol'] + df['total_ask_vol'] + epsilon)

    # 5. Differencing (Time derivatives)
    # We must be careful not to diff across sequences if df contains multiple.
    # Assuming this function might be called on a full dataset with 'seq_ix', we need groupby.
    if 'seq_ix' in df.columns:
        # Group by sequence
        # We want deltas of prices and mid-prices
        cols_to_diff = ['p0', 'p6', 'mid_price_0', 'spread_0']
        
        # We can implement a fast shift via groupby
        grouped = df.groupby('seq_ix')[cols_to_diff]
        df_diff = grouped.diff().fillna(0)
        df_diff.columns = [f'{c}_diff' for c in cols_to_diff]
        
        df = pd.concat([df, df_diff], axis=1)
    else:
        # Single sequence assumption
        cols_to_diff = ['p0', 'p6', 'mid_price_0', 'spread_0']
        for c in cols_to_diff:
            df[f'{c}_diff'] = df[c].diff().fillna(0)
            
    return df

def get_feature_columns(df):
    # Exclude metadata and targets
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']
    return [c for c in df.columns if c not in exclude]
