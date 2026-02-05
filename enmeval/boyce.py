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
    
    # Range of predictions
    fit_min, fit_max = fit.min(), fit.max()
    fit_range = fit_max - fit_min
    
    if fit_range == 0:
        return np.nan, np.array([])
    
    # Default window width is 1/10 of range
    if window_w is None:
        window_w = fit_range / 10
    
    if n_class == 0:
        # Moving window approach (matches ecospat default)
        # Create evaluation points
        eval_points = np.linspace(fit_min, fit_max, res)
        half_window = window_w / 2
        
        f_ratios = []
        valid_points = []
        
        for point in eval_points:
            # Window bounds
            lower = point - half_window
            upper = point + half_window
            
            # Count fit values in window (expected distribution)
            n_fit = np.sum((fit >= lower) & (fit < upper))
            
            # Count obs values in window (observed presences)
            n_obs = np.sum((obs >= lower) & (obs < upper))
            
            if n_fit > 0:
                # P/E ratio = (obs proportion) / (fit proportion)
                # = (n_obs/len(obs)) / (n_fit/len(fit))
                pe_ratio = (n_obs / len(obs)) / (n_fit / len(fit))
                f_ratios.append(pe_ratio)
                valid_points.append(point)
        
        f_ratios = np.array(f_ratios)
        valid_points = np.array(valid_points)
        
    else:
        # Fixed classes approach
        bin_edges = np.linspace(fit_min, fit_max, n_class + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        f_ratios = []
        valid_points = []
        
        for i in range(n_class):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            
            n_fit = np.sum((fit >= lower) & (fit < upper))
            n_obs = np.sum((obs >= lower) & (obs < upper))
            
            if n_fit > 0:
                pe_ratio = (n_obs / len(obs)) / (n_fit / len(fit))
                f_ratios.append(pe_ratio)
                valid_points.append(bin_centers[i])
        
        f_ratios = np.array(f_ratios)
        valid_points = np.array(valid_points)
    
    if len(f_ratios) < 3:
        return np.nan, f_ratios
    
    # Remove successive duplicates if requested
    if rm_duplicate and len(f_ratios) > 1:
        keep = [True]
        for i in range(1, len(f_ratios)):
            keep.append(f_ratios[i] != f_ratios[i-1])
        keep = np.array(keep)
        f_ratios = f_ratios[keep]
        valid_points = valid_points[keep]
    
    if len(f_ratios) < 3:
        return np.nan, f_ratios
    
    # Calculate correlation
    if method == 'spearman':
        cbi, _ = spearmanr(valid_points, f_ratios)
    elif method == 'pearson':
        cbi = np.corrcoef(valid_points, f_ratios)[0, 1]
    elif method == 'kendall':
        from scipy.stats import kendalltau
        cbi, _ = kendalltau(valid_points, f_ratios)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(cbi), f_ratios


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
