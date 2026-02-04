"""
Model tuning workflow for ecological niche models.

Orchestrates grid search across regularization and feature class
combinations, evaluates each with cross-validation, and returns
results for model selection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from numpy.typing import NDArray

from .partitioning import random_kfold
from .evaluation import calc_auc, calc_cbi, calc_omission_rate, calc_aicc


@dataclass
class TuningResult:
    """Results from a single model configuration."""
    regularization: float
    feature_classes: str
    auc_train: float
    auc_test: float
    cbi: float
    omission_rate: float
    aicc: Optional[float]
    n_params: int
    
    @property
    def auc_diff(self) -> float:
        """Difference between train and test AUC (overfitting indicator)."""
        return self.auc_train - self.auc_test


@dataclass  
class ENMevalResults:
    """Complete results from model tuning."""
    results: List[TuningResult]
    best_auc: TuningResult
    best_aicc: Optional[TuningResult]
    partitions: List[Tuple[NDArray, NDArray]]
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame if available."""
        try:
            import pandas as pd
            records = [
                {
                    'rm': r.regularization,
                    'fc': r.feature_classes,
                    'auc.train': r.auc_train,
                    'auc.val': r.auc_test,
                    'auc.diff': r.auc_diff,
                    'cbi': r.cbi,
                    'or.10p': r.omission_rate,
                    'AICc': r.aicc,
                    'nparams': r.n_params,
                }
                for r in self.results
            ]
            return pd.DataFrame(records)
        except ImportError:
            return self.results


def enmeval(
    presence_coords: NDArray[np.float64],
    presence_envs: NDArray[np.float64],
    background_envs: NDArray[np.float64],
    model_fn: Callable,
    regularization_multipliers: List[float] = [0.5, 1, 2, 4],
    feature_classes: List[str] = ['L', 'LQ', 'LQH', 'LQHP'],
    partitions: str = 'randomkfold',
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> ENMevalResults:
    """
    Run ENMeval model tuning.
    
    Parameters
    ----------
    presence_coords : ndarray of shape (n_presence, 2)
        Longitude, latitude of presence points (for spatial partitioning)
    presence_envs : ndarray of shape (n_presence, n_features)
        Environmental values at presence locations
    background_envs : ndarray of shape (n_background, n_features)
        Environmental values at background locations
    model_fn : callable
        Function that fits model: model_fn(train_pres, train_bg, rm, fc) -> model
        Model must have .predict(envs) method returning probabilities
    regularization_multipliers : list of float
        Regularization multiplier values to test
    feature_classes : list of str
        Feature class combinations to test (L=linear, Q=quadratic, H=hinge, P=product)
    partitions : str
        Partitioning method: 'randomkfold', 'block', 'checkerboard'
    n_folds : int
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    ENMevalResults
        Complete tuning results with best models identified
        
    Example
    -------
    >>> from elapid import MaxentModel
    >>> def fit_maxent(pres, bg, rm, fc):
    ...     model = MaxentModel(regularization=rm, feature_types=fc)
    ...     model.fit(pres, bg)
    ...     return model
    >>> results = enmeval(coords, pres_env, bg_env, fit_maxent)
    >>> print(results.best_auc)
    """
    n_presence = len(presence_envs)
    
    # Create partitions
    if partitions == 'randomkfold':
        folds = random_kfold(n_presence, k=n_folds, random_state=random_state)
    elif partitions == 'block':
        from .partitioning import block_partition
        folds = block_partition(presence_coords, k=n_folds)
    elif partitions == 'checkerboard':
        from .partitioning import checkerboard_partition
        folds = checkerboard_partition(presence_coords)
    else:
        raise ValueError(f"Unknown partition method: {partitions}")
    
    results = []
    
    # Grid search
    for rm in regularization_multipliers:
        for fc in feature_classes:
            fold_aucs_train = []
            fold_aucs_test = []
            fold_cbis = []
            fold_omissions = []
            
            for train_idx, test_idx in folds:
                # Split presence data
                train_pres = presence_envs[train_idx]
                test_pres = presence_envs[test_idx]
                
                # Fit model
                try:
                    model = model_fn(train_pres, background_envs, rm, fc)
                    
                    # Predictions
                    train_preds = model.predict(train_pres)
                    test_preds = model.predict(test_pres)
                    bg_preds = model.predict(background_envs)
                    
                    # Metrics
                    fold_aucs_train.append(calc_auc(train_preds, bg_preds))
                    fold_aucs_test.append(calc_auc(test_preds, bg_preds))
                    fold_cbis.append(calc_cbi(test_preds, bg_preds))
                    or_rate, _ = calc_omission_rate(test_preds, percentile=10)
                    fold_omissions.append(or_rate)
                    
                except Exception as e:
                    # Model failed to fit - record NaN
                    fold_aucs_train.append(np.nan)
                    fold_aucs_test.append(np.nan)
                    fold_cbis.append(np.nan)
                    fold_omissions.append(np.nan)
            
            # Average across folds
            result = TuningResult(
                regularization=rm,
                feature_classes=fc,
                auc_train=np.nanmean(fold_aucs_train),
                auc_test=np.nanmean(fold_aucs_test),
                cbi=np.nanmean(fold_cbis),
                omission_rate=np.nanmean(fold_omissions),
                aicc=None,  # TODO: implement AICc calculation
                n_params=0,  # TODO: get from model
            )
            results.append(result)
    
    # Find best models
    valid_results = [r for r in results if not np.isnan(r.auc_test)]
    best_auc = max(valid_results, key=lambda r: r.auc_test) if valid_results else results[0]
    
    # AICc-based selection (if available)
    aicc_results = [r for r in results if r.aicc is not None]
    best_aicc = min(aicc_results, key=lambda r: r.aicc) if aicc_results else None
    
    return ENMevalResults(
        results=results,
        best_auc=best_auc,
        best_aicc=best_aicc,
        partitions=folds,
    )
