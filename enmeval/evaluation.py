"""
Evaluation metrics for ecological niche models.

These metrics assess model performance using held-out test data,
with particular attention to discrimination ability and calibration.
"""

import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray


def calc_auc(
    presence_predictions: NDArray[np.float64],
    background_predictions: NDArray[np.float64]
) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Parameters
    ----------
    presence_predictions : ndarray
        Model predictions at presence (test) locations
    background_predictions : ndarray
        Model predictions at background locations
        
    Returns
    -------
    float
        AUC value between 0 and 1
        
    Notes
    -----
    AUC measures discrimination ability: the probability that
    a randomly chosen presence location has a higher predicted
    value than a randomly chosen background location.
    
    - AUC = 0.5: no better than random
    - AUC > 0.7: acceptable discrimination
    - AUC > 0.8: good discrimination
    - AUC > 0.9: excellent discrimination
    
    References
    ----------
    Fielding & Bell (1997). A review of methods for the assessment
    of prediction errors in conservation presence/absence models.
    """
    n_pres = len(presence_predictions)
    n_bg = len(background_predictions)
    
    if n_pres == 0 or n_bg == 0:
        return np.nan
    
    # Count concordant pairs using vectorized comparison
    # AUC = P(presence_pred > background_pred)
    comparisons = presence_predictions[:, np.newaxis] > background_predictions
    ties = presence_predictions[:, np.newaxis] == background_predictions
    
    auc = (comparisons.sum() + 0.5 * ties.sum()) / (n_pres * n_bg)
    return float(auc)


def calc_cbi(
    presence_predictions: NDArray[np.float64],
    background_predictions: NDArray[np.float64],
    n_bins: int = 101,
    window: int = 10
) -> float:
    """
    Calculate Continuous Boyce Index (CBI).
    
    Parameters
    ----------
    presence_predictions : ndarray
        Model predictions at presence (test) locations
    background_predictions : ndarray
        Model predictions at background locations
    n_bins : int
        Number of bins for prediction histogram (default 101)
    window : int
        Window size for moving average smoothing
        
    Returns
    -------
    float
        CBI value between -1 and 1
        
    Notes
    -----
    CBI measures how well model predictions correlate with 
    presence frequency across the prediction range.
    
    - CBI close to 1: predictions highly correlated with presences
    - CBI close to 0: model no better than random
    - CBI negative: model worse than random
    
    Unlike AUC, CBI is not affected by prevalence and focuses
    on the shape of the predicted-to-expected ratio curve.
    
    References
    ----------
    Hirzel et al. (2006). Evaluating the ability of habitat
    suitability models to predict species presences.
    
    Boyce et al. (2002). Evaluating resource selection functions.
    """
    # Create bins across the prediction range
    all_preds = np.concatenate([presence_predictions, background_predictions])
    bin_edges = np.linspace(all_preds.min(), all_preds.max(), n_bins + 1)
    
    # Count presences and expected (background) in each bin
    pres_counts, _ = np.histogram(presence_predictions, bins=bin_edges)
    bg_counts, _ = np.histogram(background_predictions, bins=bin_edges)
    
    # Normalize to proportions
    pres_freq = pres_counts / len(presence_predictions)
    bg_freq = bg_counts / len(background_predictions)
    
    # Calculate predicted-to-expected ratio (F)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        f_ratio = np.where(bg_freq > 0, pres_freq / bg_freq, np.nan)
    
    # Apply moving window smoothing
    if window > 1:
        kernel = np.ones(window) / window
        f_ratio_smooth = np.convolve(
            np.nan_to_num(f_ratio, nan=0), kernel, mode='same'
        )
    else:
        f_ratio_smooth = np.nan_to_num(f_ratio, nan=0)
    
    # CBI is the Spearman correlation between bin midpoints and F ratio
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Remove bins with no data
    valid = ~np.isnan(f_ratio) & (f_ratio_smooth > 0)
    if valid.sum() < 3:
        return np.nan
    
    # Spearman correlation
    from scipy.stats import spearmanr
    cbi, _ = spearmanr(bin_midpoints[valid], f_ratio_smooth[valid])
    
    return float(cbi)


def calc_omission_rate(
    presence_predictions: NDArray[np.float64],
    threshold: Optional[float] = None,
    percentile: float = 10.0
) -> Tuple[float, float]:
    """
    Calculate omission rate at a given threshold.
    
    Parameters
    ----------
    presence_predictions : ndarray
        Model predictions at presence (test) locations
    threshold : float, optional
        Explicit threshold value. If None, uses percentile of training data.
    percentile : float
        If threshold is None, use this percentile of predictions as threshold
        
    Returns
    -------
    tuple (omission_rate, threshold_used)
        omission_rate: proportion of presences below threshold
        threshold_used: the actual threshold value
        
    Notes
    -----
    Omission rate is the proportion of known presences that the model
    fails to predict (false negatives). Common thresholds:
    
    - 10th percentile training presence (p10): max 10% training omission
    - Minimum training presence (MTP): zero training omission
    - Equal training sensitivity and specificity (ETSS)
    
    Lower omission rates at ecologically meaningful thresholds
    indicate better model calibration.
    """
    if threshold is None:
        threshold = np.percentile(presence_predictions, percentile)
    
    omitted = presence_predictions < threshold
    omission_rate = omitted.sum() / len(presence_predictions)
    
    return float(omission_rate), float(threshold)


def calc_aicc(
    log_likelihood: float,
    n_params: int,
    n_samples: int
) -> float:
    """
    Calculate corrected Akaike Information Criterion (AICc).
    
    Parameters
    ----------
    log_likelihood : float
        Log likelihood of the model
    n_params : int
        Number of model parameters
    n_samples : int
        Number of samples used for fitting
        
    Returns
    -------
    float
        AICc value (lower is better)
        
    Notes
    -----
    AICc balances model fit against complexity, with a correction
    for small sample sizes. Used for model selection when comparing
    models with different regularization settings.
    
    AICc = AIC + (2k² + 2k) / (n - k - 1)
    where k = number of parameters, n = sample size
    
    References
    ----------
    Warren & Seifert (2011). Ecological niche modeling in Maxent:
    the importance of model complexity and the performance of
    model selection criteria.
    """
    k = n_params
    n = n_samples
    
    # Standard AIC
    aic = 2 * k - 2 * log_likelihood
    
    # Correction for small sample size
    if n - k - 1 > 0:
        correction = (2 * k * (k + 1)) / (n - k - 1)
    else:
        correction = np.inf
    
    return float(aic + correction)
