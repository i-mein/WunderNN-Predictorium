"""
Utilities for exp_09.
Copied from exp_06 for compatibility.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import pearsonr


@dataclass
class DataPoint:
    """Single data point for prediction."""
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class ScorerStepByStep:
    """
    Scorer that simulates step-by-step prediction.
    Matches the evaluation environment.
    """
    
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.targets = ['t0', 't1']
        
    def score(self, model):
        """
        Score a model on the dataset.
        
        Args:
            model: Model with predict(DataPoint) method
        
        Returns:
            Dictionary with scores
        """
        # Collect predictions
        predictions = {t: [] for t in self.targets}
        actuals = {t: [] for t in self.targets}
        
        # Raw feature columns
        raw_cols = (
            [f'p{i}' for i in range(12)] +
            [f'v{i}' for i in range(12)] +
            [f'dp{i}' for i in range(4)] +
            [f'dv{i}' for i in range(4)]
        )
        
        # Iterate through sequences
        for seq_ix in sorted(self.df['seq_ix'].unique()):
            seq_df = self.df[self.df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            
            for _, row in seq_df.iterrows():
                state = row[raw_cols].values.astype(np.float32)
                
                data_point = DataPoint(
                    seq_ix=row['seq_ix'],
                    step_in_seq=row['step_in_seq'],
                    need_prediction=row['need_prediction'],
                    state=state
                )
                
                pred = model.predict(data_point)
                
                if data_point.need_prediction and pred is not None:
                    predictions['t0'].append(pred[0])
                    predictions['t1'].append(pred[1])
                    actuals['t0'].append(row['t0'])
                    actuals['t1'].append(row['t1'])
        
        # Calculate correlations
        results = {}
        for target in self.targets:
            if len(predictions[target]) > 0:
                corr, _ = pearsonr(predictions[target], actuals[target])
                results[target] = corr
            else:
                results[target] = 0.0
        
        # Weighted average (equal weights for t0 and t1)
        results['weighted_pearson'] = (results['t0'] + results['t1']) / 2
        
        return results
