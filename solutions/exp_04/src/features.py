import numpy as np
import pandas as pd

def generate_features(df_seq):
    """
    Exp 04 Features: Stationary Technical Indicators.
    """
    df = df_seq.copy()
    
    # --- PHASE 1: Base (Exp 02) ---
    df['mid_price_0'] = (df['p0'] + df['p6']) / 2
    df['mid_price_1'] = (df['p1'] + df['p7']) / 2
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

    # --- PHASE 2: Technical Indicators (Stationary) ---
    # Calc on [mid_price_0, spread_0]
    target_cols = ['mid_price_0', 'spread_0']
    
    if 'seq_ix' in df.columns:
        grouped = df.groupby('seq_ix')[target_cols]
        # RSI (14) - Cutler's (Simple Moving Average of Gains/Losses)
        # 1. Delta
        # We need to process group by group or use transformation
        # RSI is tricky to vectorize purely with groupby().rolling() because of gain/loss split.
        
        # However, we can calculate Gain/Loss columns first (diff is safe within seq if handled)
        # But diff across seq boundary is bad.
        # Grouped diff:
        
        # Optimized approach:
        # Calculate diffs for all, mask out boundaries? 
        # Safest: Grouped apply? Slow.
        # groupby().diff() is fast.
        
        diffs = grouped.diff().fillna(0) # (N, 2)
        
        for col in target_cols:
            delta = diffs[col]
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            # Helper for rolling mean with groupby
            # We must regroup the gains/losses
            # Actually, `diffs` index aligns with `df`.
            # We can map `seq_ix` to these new series and group.
            
            # Create temp df for rolling calc
            temp = pd.DataFrame({'g': gain, 'l': loss, 'seq': df['seq_ix']})
            g_roll = temp.groupby('seq')['g'].rolling(14, min_periods=1).mean().reset_index(0, drop=True)
            l_roll = temp.groupby('seq')['l'].rolling(14, min_periods=1).mean().reset_index(0, drop=True)
            
            rs = g_roll / (l_roll + epsilon)
            df[f'{col}_rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands %B (20)
            # Needs Moving Average and Std of Price (not diffs)
            
            # Re-get grouped price
            price_grouped = df.groupby('seq_ix')[col]
            ma = price_grouped.rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            std = price_grouped.rolling(20, min_periods=1).std().fillna(0).reset_index(0, drop=True)
            
            # %B = (Price - Lower) / (Upper - Lower)
            # Upper = ma + 2std
            # Lower = ma - 2std
            # Diff = 4std
            # %B = (Price - ma + 2std) / 4std
            
            # We use original series `df[col]` for Price
            # Indexes align if reset_index(drop=True) works correctly with sorted groups.
            # Assuming df is sorted by seq_ix, step_in_seq.
            
            pb = (df[col] - ma + 2*std) / (4*std + epsilon)
            df[f'{col}_bb_pband_20'] = pb
            
    else:
        # Single sequence
        diffs = df[target_cols].diff().fillna(0)
        for col in target_cols:
            delta = diffs[col]
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(14, min_periods=1).mean()
            avg_loss = loss.rolling(14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + epsilon)
            df[f'{col}_rsi_14'] = 100 - (100 / (1 + rs))
            
            ma = df[col].rolling(20, min_periods=1).mean()
            std = df[col].rolling(20, min_periods=1).std().fillna(0)
            df[f'{col}_bb_pband_20'] = (df[col] - ma + 2*std) / (4*std + epsilon)

    # --- PHASE 3: Diffs ---
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
