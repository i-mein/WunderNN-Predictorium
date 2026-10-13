"""
Feature extraction for exp_10: Raw features only

Key insight from experiments:
- exp_01 (32 raw features) had BEST LB score (0.2820)
- exp_06 (65 engineered features) had lower LB (0.2634)
- Feature engineering may hurt generalization

Therefore: Use only raw 32 features like exp_01.
"""

import numpy as np


# Raw feature names (32 total)
RAW_FEATURES = [
    # Prices at different levels (12)
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11',
    # Volumes at different levels (12)
    'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
    # Price differences (4)
    'dp0', 'dp1', 'dp2', 'dp3',
    # Volume differences (4)
    'dv0', 'dv1', 'dv2', 'dv3',
]

# Target names
TARGETS = ['t0', 't1']

# Need prediction mask
MASK_COLUMN = 'need_prediction'


def extract_raw_features(df):
    """
    Extract raw 32 features from dataframe.
    
    Args:
        df: DataFrame with LOB data
    
    Returns:
        features: numpy array (n_samples, 32)
        targets: numpy array (n_samples, 2) for t0, t1
        mask: numpy array (n_samples,) boolean for need_prediction
    """
    # Extract features
    features = df[RAW_FEATURES].values.astype(np.float32)
    
    # Extract targets
    targets = df[TARGETS].values.astype(np.float32)
    
    # Extract mask
    mask = df[MASK_COLUMN].values.astype(bool)
    
    return features, targets, mask


def normalize_features(features, mean=None, std=None):
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: numpy array (n_samples, n_features)
        mean: Optional precomputed mean (for test-time normalization)
        std: Optional precomputed std
    
    Returns:
        normalized: Normalized features
        mean: Feature means
        std: Feature stds
    """
    if mean is None:
        mean = np.mean(features, axis=0, keepdims=True)
    if std is None:
        std = np.std(features, axis=0, keepdims=True) + 1e-8
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def prepare_sequences(features, targets, mask, seq_length=1000):
    """
    Prepare sequences for training.
    
    Args:
        features: numpy array (n_samples, n_features)
        targets: numpy array (n_samples, 2)
        mask: numpy array (n_samples,) boolean
        seq_length: Sequence length
    
    Returns:
        X: Sequences (n_sequences, seq_length, n_features)
        y: Targets (n_sequences, seq_length, 2)
        m: Masks (n_sequences, seq_length)
    """
    n_samples = len(features)
    n_sequences = n_samples // seq_length
    
    # Truncate to full sequences
    n_samples_used = n_sequences * seq_length
    
    X = features[:n_samples_used].reshape(n_sequences, seq_length, -1)
    y = targets[:n_samples_used].reshape(n_sequences, seq_length, -1)
    m = mask[:n_samples_used].reshape(n_sequences, seq_length)
    
    return X, y, m


class LOBFeatureExtractor:
    """
    Feature extractor for LOB data.
    
    Handles normalization and sequence preparation.
    """
    
    def __init__(self, seq_length=1000):
        self.seq_length = seq_length
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, df):
        """
        Fit normalization parameters on training data.
        
        Args:
            df: Training DataFrame
        """
        features, _, _ = extract_raw_features(df)
        _, self.mean, self.std = normalize_features(features)
        self.fitted = True
    
    def transform(self, df):
        """
        Transform data to sequences.
        
        Args:
            df: DataFrame with LOB data
        
        Returns:
            X: Feature sequences
            y: Target sequences
            m: Mask sequences
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        features, targets, mask = extract_raw_features(df)
        features_norm, _, _ = normalize_features(features, self.mean, self.std)
        X, y, m = prepare_sequences(features_norm, targets, mask, self.seq_length)
        
        return X, y, m
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_normalization_params(self):
        """Get normalization parameters for inference."""
        return {
            'mean': self.mean,
            'std': self.std,
        }


def test_features():
    """Test feature extraction."""
    import pandas as pd
    
    # Create dummy data
    n_samples = 5000
    data = {}
    
    for feat in RAW_FEATURES:
        data[feat] = np.random.randn(n_samples).astype(np.float32)
    
    for target in TARGETS:
        data[target] = np.random.randn(n_samples).astype(np.float32)
    
    data[MASK_COLUMN] = np.random.randint(0, 2, n_samples).astype(int)
    
    df = pd.DataFrame(data)
    
    # Test extraction
    extractor = LOBFeatureExtractor(seq_length=1000)
    X, y, m = extractor.fit_transform(df)
    
    print(f"Features: {len(RAW_FEATURES)} total")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Mask shape: {m.shape}")
    print(f"Mask sum (valid samples): {m.sum()}")
    
    # Check normalization
    print(f"\nNormalization check:")
    print(f"  Mean of features: {X.mean():.6f} (should be ~0)")
    print(f"  Std of features: {X.std():.6f} (should be ~1)")


if __name__ == "__main__":
    test_features()
