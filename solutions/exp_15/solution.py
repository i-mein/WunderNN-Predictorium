"""
Solution for exp_15: Enhanced CatBoost.

Inference-only code for submission.
Uses NumPy-based feature generation including micro-price and trade flow.
"""
import numpy as np
from catboost import CatBoostRegressor


class CatBoostSolution:
    """
    Enhanced CatBoost solution with micro-price and trade flow features.
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
        self.micro_price_deviation_buffer = []
        self.depth_imbal_near_buffer = []
        
        # Previous values for diff computation
        self.prev_values = {}
    
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
            features: numpy array of all features for this step
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
        # 3. MICRO-PRICE FEATURES (5)
        # ========================================
        
        # Micro-price
        micro_price = (p[0] * v[6] + p[6] * v[0]) / (v[0] + v[6] + self.epsilon)
        features.append(micro_price)
        
        # Micro-price deviation
        micro_price_deviation = micro_price - mid_price_0
        features.append(micro_price_deviation)
        
        # Normalized micro-price deviation
        micro_price_deviation_norm = micro_price_deviation / (spread_0 + self.epsilon)
        features.append(micro_price_deviation_norm)
        
        # Weighted mid-price level 2
        weighted_mid_2 = (p[1] * v[7] + p[7] * v[1]) / (v[1] + v[7] + self.epsilon)
        features.append(weighted_mid_2)
        
        # Level 2 micro-price deviation
        micro_price_deviation_2 = weighted_mid_2 - mid_price_1
        features.append(micro_price_deviation_2)
        
        # ========================================
        # 4. ORDER BOOK PRESSURE (12)
        # ========================================
        
        # Bid/Ask pressure
        bid_pressure_0 = v[0] / (total_bid_vol + self.epsilon)
        bid_pressure_01 = (v[0] + v[1]) / (total_bid_vol + self.epsilon)
        ask_pressure_0 = v[6] / (total_ask_vol + self.epsilon)
        ask_pressure_01 = (v[6] + v[7]) / (total_ask_vol + self.epsilon)
        features.extend([bid_pressure_0, bid_pressure_01, ask_pressure_0, ask_pressure_01])
        
        # LOB slope
        bid_vwap = sum(p[i] * v[i] for i in range(6)) / (total_bid_vol + self.epsilon)
        ask_vwap = sum(p[i+6] * v[i+6] for i in range(6)) / (total_ask_vol + self.epsilon)
        
        bid_slope = (p[0] - bid_vwap) / (total_bid_vol + self.epsilon)
        ask_slope = (ask_vwap - p[6]) / (total_ask_vol + self.epsilon)
        slope_imbal = (bid_slope - ask_slope) / (bid_slope + ask_slope + self.epsilon)
        features.extend([bid_slope, ask_slope, slope_imbal])
        
        # Depth imbalance
        depth_imbal_near = (v[0] + v[1] - v[6] - v[7]) / \
                           (v[0] + v[1] + v[6] + v[7] + self.epsilon)
        depth_imbal_far = (v[4] + v[5] - v[10] - v[11]) / \
                          (v[4] + v[5] + v[10] + v[11] + self.epsilon)
        features.extend([depth_imbal_near, depth_imbal_far])
        
        # VWAP features
        features.extend([bid_vwap, ask_vwap, ask_vwap - bid_vwap])
        
        # ========================================
        # 5. PRICE LEVEL FEATURES (6)
        # ========================================
        
        price_range = p[11] - p[0]
        bid_range = p[5] - p[0]
        ask_range = p[11] - p[6]
        bid_spacing = (p[0] - p[5]) / 5
        ask_spacing = (p[11] - p[6]) / 5
        spacing_ratio = bid_spacing / (ask_spacing + self.epsilon)
        features.extend([price_range, bid_range, ask_range, bid_spacing, ask_spacing, spacing_ratio])
        
        # ========================================
        # 6. ROLLING FEATURES (20)
        # ========================================
        
        # Update buffers
        self.mid_price_0_buffer.append(mid_price_0)
        self.spread_0_buffer.append(spread_0)
        self.vol_imbal_total_buffer.append(vol_imbal_total)
        self.micro_price_deviation_buffer.append(micro_price_deviation)
        self.depth_imbal_near_buffer.append(depth_imbal_near)
        
        # Compute rolling stats
        features.extend(self._compute_rolling_stats(self.mid_price_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.spread_0_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.vol_imbal_total_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.micro_price_deviation_buffer, self.windows))
        features.extend(self._compute_rolling_stats(self.depth_imbal_near_buffer, self.windows))
        
        # ========================================
        # 7. DIFF FEATURES (8)
        # ========================================
        
        diff_keys = ['p0', 'p6', 'mid_price_0', 'spread_0', 'vol_imbal_total',
                     'micro_price', 'micro_price_deviation', 'depth_imbal_near']
        diff_values = [p[0], p[6], mid_price_0, spread_0, vol_imbal_total,
                       micro_price, micro_price_deviation, depth_imbal_near]
        
        if self.step == 0:
            features.extend([0.0] * len(diff_keys))
        else:
            for key, val in zip(diff_keys, diff_values):
                features.append(val - self.prev_values.get(key, val))
        
        # Update previous values
        for key, val in zip(diff_keys, diff_values):
            self.prev_values[key] = val
        
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
