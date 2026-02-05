"""
Continuous Boyce Index (CBI) implementation.

Matches ecospat.boyce() from the R ecospat package.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr


def calc_boyce(
    fit: NDArray[np.float64],
    obs: NDArray[np.float64],
    n_class: int = 0,
    window_w: float = None,
    res: int = 100,
    rm_duplicate: bool = True,
    method: str = 'spearman'
) -> tuple[float, NDArray[np.float64]]:
    """
    Calculate Continuous Boyce Index as in Hirzel et al. (2006).
    
    Matches ecospat.boyce() from R's ecospat package.
    
    Parameters
    ----------
    fit : ndarray
        Predicted suitability values for all locations (presence + background)
    obs : ndarray
        Predicted suitability values at presence (validation) locations
    n_class : int
        Number of classes. If 0, uses moving window approach (default)
    window_w : float
        Width of moving window. Default is 1/10 of suitability range
    res : int
        Resolution (number of evaluation points). Default 100
    rm_duplicate : bool
        If True, remove successive duplicate P/E values (default True)
    method : str
        Correlation method: 'spearman', 'pearson', or 'kendall'
        
    Returns
    -------
    tuple[float, ndarray]
        (boyce_index, f_ratios) - the CBI value and P/E ratios
        
    References
    ----------
    Boyce et al. (2002). Evaluating resource selection functions.
    Hirzel et al. (2006). Evaluating habitat suitability models.
    """
    fit = np.asarray(fit).flatten()
    obs = np.asarray(obs).flatten()
    
    # R uses combined min/max of fit and obs
    mini = min(fit.min(), obs.min())
    maxi = max(fit.max(), obs.max())
    
    if maxi - mini == 0:
        return np.nan, np.array([])
    
    # Default window width is 1/10 of range
    if window_w is None:
        window_w = (fit.max() - fit.min()) / 10
    
    def boycei(interval, obs, fit):
        """Calculate P/E ratio for an interval (matches R)."""
        pi = np.sum((obs >= interval[0]) & (obs <= interval[1])) / len(obs)
        ei = np.sum((fit >= interval[0]) & (fit <= interval[1])) / len(fit)
        if ei == 0:
            return np.nan
        return round(pi / ei, 10)
    
    if n_class == 0:
        # Moving window approach (matches R exactly)
        vec_mov = np.linspace(mini, maxi - window_w, res + 1)
        vec_mov[-1] = vec_mov[-1] + 1  # R adds 1 to last element
        interval = np.column_stack([vec_mov, vec_mov + window_w])
    else:
        # Fixed classes approach
        vec_mov = np.linspace(mini, maxi, n_class + 1)
        interval = np.column_stack([vec_mov[:-1], vec_mov[1:]])
    
    # Calculate F ratios
    f = np.array([boycei(intv, obs, fit) for intv in interval])
    
    # Track which values to keep (non-NaN)
    to_keep = ~np.isnan(f)
    f_kept = f[to_keep]
    vec_kept = vec_mov[to_keep] if n_class == 0 else (interval[to_keep, 0] + interval[to_keep, 1]) / 2
    
    if len(f_kept) < 3:
        return np.nan, f
    
    # Remove successive duplicates if requested (matches R)
    if rm_duplicate and len(f_kept) > 1:
        # R: r <- c(1:length(f))[f != c(f[-1], TRUE)]
        shifted = np.append(f_kept[1:], np.nan)  # shift left, pad with nan
        r = f_kept != shifted
        f_corr = f_kept[r]
        vec_corr = vec_kept[r]
    else:
        f_corr = f_kept
        vec_corr = vec_kept
    
    if len(f_corr) < 3:
        return np.nan, f
    
    # Calculate correlation
    if method == 'spearman':
        cbi, _ = spearmanr(f_corr, vec_corr)
    elif method == 'pearson':
        cbi = np.corrcoef(f_corr, vec_corr)[0, 1]
    elif method == 'kendall':
        from scipy.stats import kendalltau
        cbi, _ = kendalltau(f_corr, vec_corr)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(cbi), f


# Convenience wrapper matching my original API
def calc_cbi(
    presence_predictions: NDArray[np.float64],
    background_predictions: NDArray[np.float64],
    **kwargs
) -> float:
    """
    Calculate CBI from presence and background predictions.
    
    This is a convenience wrapper around calc_boyce() that takes
    separate presence and background predictions.
    """
    # Combine to get 'fit' (all predictions)
    fit = np.concatenate([presence_predictions, background_predictions])
    # 'obs' is just the presence predictions
    obs = presence_predictions
    
    cbi, _ = calc_boyce(fit, obs, **kwargs)
    return cbi
