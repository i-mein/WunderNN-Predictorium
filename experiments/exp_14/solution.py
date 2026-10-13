"""
Solution for exp_14: CatBoost Baseline.

Inference-only code for submission.
Uses NumPy-based feature generation for speed.
"""
import numpy as np
from catboost import CatBoostRegressor


class CatBoostSolution:
    """
    CatBoost-based solution with incremental feature computation.
    """
    
    def __init__(self):
        # Load pre-trained models
        self.model_t0 = CatBoostRegressor()
        self.model_t1 = CatBoostRegressor()
        self.model_t0.load_model('model_catboost_t0.cbm')
        self.model_t1.load_model('model_catboost_t1.cbm')
        
        # Rolling state for incremental computation
        self.reset()
        
        # Constants
        self.epsilon = 1e-8
        self.windows = [3, 10]
        
    def reset(self):
        """Reset state for new sequence."""
        self.step = 0
        # Rolling buffers for features that need history
        self.mid_price_0_buffer = []
        self.spread_0_buffer = []
        self.vol_imbal_total_buffer = []
        # Previous values for diff computation
        self.prev_p0 = None
        self.prev_p6 = None
        self.prev_mid_price_0 = None
        self.prev_spread_0 = None
        self.prev_vol_imbal_total = None
    
    def _compute_rolling_stats(self, buffer, windows):
        """Compute rolling mean and std for given windows."""
        stats = []
        arr = np.array(buffer)
        for w in windows:
            if len(arr) >= 1:
                window_data = arr[-w:] if len(arr) >= w else arr
                stats.append(np.mean(window_data))
                stats.append(np.std(window_data) if len(window_data) > 1 else 0.0)
            else:
                stats.extend([0.0, 0.0])
        return stats
    
    def generate_features(self, step_data):
        """
        Generate features for a single timestep.
        
        Args:
            step_data: numpy array of shape (24,) containing:
                - p0-p5: bid prices (6)
                - p6-p11: ask prices (6)
                - v0-v5: bid volumes (6)
                - v6-v11: ask volumes (6)
        
        Returns:
            features: numpy array of shape (65,) - all features for this step
        """
        # Unpack raw features
        p = step_data[:12]   # Prices
        v = step_data[12:24] # Volumes
        
        features = []
        
        # ========================================
        # 1. RAW FEATURES (24)
        # ========================================
        features.extend(p)  # p0-p11
        features.extend(v)  # v0-v11
        
        # ========================================
        # 2. BASE ENGINEERED FEATURES (16)
        # ========================================
        
        # Mid prices
        mid_price_0 = (p[0] + p[6]) * 0.5
        mid_price_1 = (p[1] + p[7]) * 0.5
        features.append(mid_price_0)
        features.append(mid_price_1)
        
        # Spreads
        spread_0 = p[6] - p[0]
        spread_1 = p[7] - p[1]
        features.append(spread_0)
        features.append(spread_1)
        
        # Volume imbalances per level (6)
        for i in range(6):
            bv = v[i]
            av = v[i + 6]
            vol_imbal = (bv - av) / (bv + av + self.epsilon)
            features.append(vol_imbal)
        
        # Total volumes and imbalance
        total_bid_vol = sum(v[:6])
        total_ask_vol = sum(v[6:])
        features.append(total_bid_vol)
        features.append(total_ask_vol)
        
        vol_imbal_total = (total_bid_vol - total_ask_vol) / \
                          (total_bid_vol + total_ask_vol + self.epsilon)
        features.append(vol_imbal_total)
        
        # Relative spread
        relative_spread_0 = spread_0 / (mid_price_0 + self.epsilon)
        features.append(relative_spread_0)
        
        # Volume ratio
        vol_ratio = total_bid_vol / (total_ask_vol + self.epsilon)
        features.append(vol_ratio)
        
        # Price depth
        price_depth = (p[5] - p[0]) / (p[0] + self.epsilon)
        features.append(price_depth)
        
        # ========================================
        # 3. ROLLING FEATURES (12)
        # ========================================
        
        # Update buffers
        self.mid_price_0_buffer.append(mid_price_0)
        self.spread_0_buffer.append(spread_0)
        self.vol_imbal_total_buffer.append(vol_imbal_total)
        
        # Compute rolling stats
        features.extend(self._compute_rolling_stats(self.mid_price_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.spread_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.vol_imbal_total_buffer, self.windows))
        
        # ========================================
        # 4. DIFF FEATURES (5)
        # ========================================
        
        # First step: diffs are 0
        if self.step == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.append(p[0] - self.prev_p0)           # p0_diff
            features.append(p[6] - self.prev_p6)           # p6_diff
            features.append(mid_price_0 - self.prev_mid_price_0)  # mid_price_0_diff
            features.append(spread_0 - self.prev_spread_0)        # spread_0_diff
            features.append(vol_imbal_total - self.prev_vol_imbal_total)  # vol_imbal_total_diff
        
        # Update previous values
        self.prev_p0 = p[0]
        self.prev_p6 = p[6]
        self.prev_mid_price_0 = mid_price_0
        self.prev_spread_0 = spread_0
        self.prev_vol_imbal_total = vol_imbal_total
        
        self.step += 1
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, step_data):
        """
        Predict t0 and t1 for a single timestep.
        
        Args:
            step_data: numpy array of shape (24,)
        
        Returns:
            predictions: numpy array of shape (2,) - [t0_pred, t1_pred]
        """
        # Generate features
        features = self.generate_features(step_data)
        
        # Reshape for CatBoost (expects 2D array)
        features_2d = features.reshape(1, -1)
        
        # Predict
        t0_pred = self.model_t0.predict(features_2d)[0]
        t1_pred = self.model_t1.predict(features_2d)[0]
        
        return np.array([t0_pred, t1_pred], dtype=np.float32)


# Global solution instance
_solution = None


def reset():
    """Reset the solution state for a new sequence."""
    global _solution
    if _solution is None:
        _solution = CatBoostSolution()
    _solution.reset()


def predict(step_data):
    """
    Predict for one timestep.
    
    Args:
        step_data: numpy array of shape (24,) or (1, 24)
    
    Returns:
        predictions: numpy array of shape (2,) - [t0_pred, t1_pred]
    """
    global _solution
    if _solution is None:
        _solution = CatBoostSolution()
    
    # Handle shape
    if len(step_data.shape) > 1:
        step_data = step_data.flatten()
    
    return _solution.predict(step_data)
