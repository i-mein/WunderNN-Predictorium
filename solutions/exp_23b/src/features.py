"""
exp_23: Enhanced LOB Feature Engineering

Features:
- 32 raw features (p0-p11, v0-v11, dp0-dp3, dv0-dv3)
- 15 enhanced features:
  * Micro-price (volume-weighted mid)
  * Order Flow Imbalance (OFI)
  * Weighted price pressure
  * Spread features
  * Volume imbalance features
  * Price momentum

Total: 47 features
"""

import numpy as np
import warnings

# Suppress numpy warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def safe_divide(a, b, default=0.0):
    """Safe division that handles zero denominators."""
    result = np.where(np.abs(b) > 1e-10, a / b, default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


def safe_log(x, default=0.0):
    """Safe log that handles non-positive values."""
    x_safe = np.clip(x, 1e-10, 1e10)
    result = np.log(x_safe)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


def compute_enhanced_features(state: np.ndarray) -> np.ndarray:
    """
    Compute enhanced LOB features from raw 32-feature state.
    
    Raw features layout (32 total):
    - p0-p5: bid prices (level 0-5)
    - p6-p11: ask prices (level 0-5)  
    - v0-v5: bid volumes (level 0-5)
    - v6-v11: ask volumes (level 0-5)
    - dp0-dp3: delta prices (4)
    - dv0-dv3: delta volumes (4)
    
    Returns:
        Enhanced feature vector (47,) = 32 raw + 15 enhanced
    """
    # Handle NaN in input
    s = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract raw components
    # Prices: p0-p5 are bids (descending), p6-p11 are asks (ascending)
    p_bid = s[0:6]   # Bid prices, best to worst
    p_ask = s[6:12]  # Ask prices, best to worst
    v_bid = np.maximum(s[12:18], 0.0)  # Bid volumes (ensure non-negative)
    v_ask = np.maximum(s[18:24], 0.0)  # Ask volumes (ensure non-negative)
    dp = s[24:28]    # Delta prices
    dv = s[28:32]    # Delta volumes
    
    # Best bid/ask
    best_bid = p_bid[0]
    best_ask = p_ask[0]
    best_bid_vol = max(v_bid[0], 0.0)
    best_ask_vol = max(v_ask[0], 0.0)
    
    # === Enhanced Features (15 total) ===
    
    # 1. Mid-price (standard)
    mid_price = (best_bid + best_ask) * 0.5
    
    # 2. Micro-price (volume-weighted mid-price)
    total_best_vol = best_bid_vol + best_ask_vol
    if total_best_vol > 1e-10:
        micro_price = (best_ask * best_bid_vol + best_bid * best_ask_vol) / total_best_vol
    else:
        micro_price = mid_price
    
    # 3. Spread
    spread = best_ask - best_bid
    relative_spread = safe_divide(spread, abs(mid_price) + 1e-8)
    
    # 4. Volume imbalance at best level
    vol_sum_0 = best_bid_vol + best_ask_vol
    vol_imbal_0 = safe_divide(best_bid_vol - best_ask_vol, vol_sum_0 + 1e-8)
    
    # 5. Total volume imbalance (all levels)
    total_bid_vol = np.sum(v_bid)
    total_ask_vol = np.sum(v_ask)
    total_vol = total_bid_vol + total_ask_vol
    vol_imbal_total = safe_divide(total_bid_vol - total_ask_vol, total_vol + 1e-8)
    
    # 6. Weighted Price Pressure (WPP)
    wpp = 0.0
    weights = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    for i in range(6):
        vol_sum = v_bid[i] + v_ask[i]
        if vol_sum > 1e-10:
            level_imbal = (v_bid[i] - v_ask[i]) / vol_sum
            wpp += weights[i] * level_imbal
    
    # 7. Order Flow Imbalance (OFI)
    ofi = dv[0] - dv[1]
    
    # 8. Price momentum (sum of delta prices)
    price_momentum = np.sum(dp)
    
    # 9. Volume momentum (sum of delta volumes)
    vol_momentum = np.sum(dv)
    
    # 10. Book depth ratio (log scale)
    if total_ask_vol > 1e-10:
        depth_ratio = total_bid_vol / total_ask_vol
    else:
        depth_ratio = 1.0
    log_depth_ratio = safe_log(depth_ratio)
    
    # 11. Price depth (how far is level 5 from best)
    bid_depth = safe_divide(best_bid - p_bid[5], abs(best_bid) + 1e-8)
    ask_depth = safe_divide(p_ask[5] - best_ask, abs(best_ask) + 1e-8)
    
    # 12. Mid-price vs Micro-price difference
    mid_micro_diff = micro_price - mid_price
    
    # 13. Normalized spread (spread / mid)
    norm_spread = safe_divide(spread, abs(mid_price) + 1e-8)
    
    # Compile enhanced features (15)
    enhanced = np.array([
        mid_price,           # 1
        micro_price,         # 2
        spread,              # 3
        relative_spread,     # 4
        vol_imbal_0,         # 5
        vol_imbal_total,     # 6
        wpp,                 # 7
        ofi,                 # 8
        price_momentum,      # 9
        vol_momentum,        # 10
        log_depth_ratio,     # 11
        bid_depth,           # 12
        ask_depth,           # 13
        mid_micro_diff,      # 14
        norm_spread,         # 15
    ], dtype=np.float32)
    
    # Final NaN cleanup for enhanced features
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Concatenate: 32 raw + 15 enhanced = 47 features
    result = np.concatenate([s, enhanced])
    
    # Final safety check - replace any remaining NaN/Inf
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result.astype(np.float32)


class FeatureEngineer:
    """
    Stateful feature engineer for step-by-step inference.
    Maintains history for any rolling features (not used currently to avoid leakage).
    """
    
    def __init__(self):
        self.current_seq_ix = None
        self.prev_state = None
    
    def reset(self):
        """Reset state for new sequence."""
        self.prev_state = None
    
    def compute(self, state: np.ndarray, seq_ix: int = None) -> np.ndarray:
        """
        Compute features for a single timestep.
        
        Args:
            state: Raw 32-feature state
            seq_ix: Sequence ID (for state reset)
        
        Returns:
            47-feature vector
        """
        # Reset on new sequence
        if seq_ix is not None and seq_ix != self.current_seq_ix:
            self.current_seq_ix = seq_ix
            self.reset()
        
        # Compute enhanced features
        features = compute_enhanced_features(state)
        
        # Update state
        self.prev_state = state.copy()
        
        return features


def compute_features_batch(states: np.ndarray) -> np.ndarray:
    """
    Compute enhanced features for a batch of sequences.
    
    Args:
        states: (batch, seq_len, 32) raw features
    
    Returns:
        (batch, seq_len, 47) enhanced features
    """
    batch_size, seq_len, _ = states.shape
    
    # Pre-allocate output
    output = np.zeros((batch_size, seq_len, 47), dtype=np.float32)
    
    for b in range(batch_size):
        for t in range(seq_len):
            output[b, t] = compute_enhanced_features(states[b, t])
    
    return output


if __name__ == "__main__":
    # Test feature computation
    np.random.seed(42)
    
    # Simulate a single state
    state = np.random.randn(32).astype(np.float32)
    state[0:12] = np.abs(state[0:12]) * 100 + 100  # Prices should be positive
    state[12:24] = np.abs(state[12:24]) * 1000     # Volumes should be positive
    
    features = compute_enhanced_features(state)
    print(f"Input shape: {state.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Raw features: {features[:32].shape}")
    print(f"Enhanced features: {features[32:].shape}")
    
    print("\nEnhanced feature values:")
    names = [
        "mid_price", "micro_price", "spread", "relative_spread",
        "vol_imbal_0", "vol_imbal_total", "wpp", "ofi",
        "price_momentum", "vol_momentum", "log_depth_ratio",
        "bid_depth", "ask_depth", "mid_micro_diff", "norm_spread"
    ]
    for i, name in enumerate(names):
        print(f"  {name}: {features[32+i]:.4f}")
    
    # Test batch computation
    print("\nTesting batch computation...")
    batch = np.random.randn(4, 100, 32).astype(np.float32)
    batch[:, :, 0:12] = np.abs(batch[:, :, 0:12]) * 100 + 100
    batch[:, :, 12:24] = np.abs(batch[:, :, 12:24]) * 1000
    
    batch_features = compute_features_batch(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {batch_features.shape}")
