"""
Spatial and non-spatial partitioning methods for cross-validation.

These methods split occurrence data into training and testing folds
for model evaluation, with special attention to spatial autocorrelation.
"""

import numpy as np
from typing import List, Tuple, Optional
from numpy.typing import NDArray


def random_kfold(
    n_samples: int,
    k: int = 5,
    random_state: Optional[int] = None
) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Random k-fold partitioning.
    
    Parameters
    ----------
    n_samples : int
        Number of occurrence records
    k : int
        Number of folds (default 5)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    List of (train_indices, test_indices) tuples
    
    Notes
    -----
    Standard k-fold CV. Does not account for spatial autocorrelation.
    Use spatial methods (block, checkerboard) when occurrences are clustered.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_idx, test_idx))
        current += fold_size
    
    return folds


def leave_one_out(n_samples: int) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Leave-one-out (jackknife) partitioning.
    
    Parameters
    ----------
    n_samples : int
        Number of occurrence records
        
    Returns
    -------
    List of (train_indices, test_indices) tuples
    
    Notes
    -----
    Each fold uses one sample for testing, rest for training.
    Computationally expensive for large datasets.
    """
    indices = np.arange(n_samples)
    folds = []
    for i in range(n_samples):
        test_idx = np.array([i])
        train_idx = np.concatenate([indices[:i], indices[i+1:]])
        folds.append((train_idx, test_idx))
    return folds


def block_partition(
    coords: NDArray[np.float64],
    k: int = 4,
    orientation: str = "auto"
) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Spatial block partitioning.
    
    Divides geographic space into k contiguous blocks to account
    for spatial autocorrelation. Points in the same block are
    never split between training and testing.
    
    Parameters
    ----------
    coords : ndarray of shape (n_samples, 2)
        Longitude, latitude coordinates
    k : int
        Number of blocks (default 4)
    orientation : str
        How to divide: "lat" (horizontal), "lon" (vertical), 
        "auto" (alternates for squarish blocks)
        
    Returns
    -------
    List of (train_indices, test_indices) tuples
    
    References
    ----------
    Muscarella et al. (2014). ENMeval: An R package for conducting
    spatially independent evaluations.
    """
    n_samples = len(coords)
    
    # Determine block assignments based on coordinate quantiles
    if orientation == "lat" or (orientation == "auto" and k <= 2):
        # Divide by latitude
        quantiles = np.quantile(coords[:, 1], np.linspace(0, 1, k + 1))
        assignments = np.digitize(coords[:, 1], quantiles[1:-1])
    elif orientation == "lon":
        # Divide by longitude
        quantiles = np.quantile(coords[:, 0], np.linspace(0, 1, k + 1))
        assignments = np.digitize(coords[:, 0], quantiles[1:-1])
    else:
        # Auto: create roughly square blocks
        # First split by lon, then by lat within each
        n_lon = int(np.ceil(np.sqrt(k)))
        n_lat = int(np.ceil(k / n_lon))
        
        lon_quantiles = np.quantile(coords[:, 0], np.linspace(0, 1, n_lon + 1))
        lon_bins = np.digitize(coords[:, 0], lon_quantiles[1:-1])
        
        assignments = np.zeros(n_samples, dtype=int)
        for lon_bin in range(n_lon):
            mask = lon_bins == lon_bin
            if mask.sum() > 0:
                lat_vals = coords[mask, 1]
                lat_quantiles = np.quantile(lat_vals, np.linspace(0, 1, n_lat + 1))
                lat_bins = np.digitize(lat_vals, lat_quantiles[1:-1])
                assignments[mask] = lon_bin * n_lat + lat_bins
    
    # Create folds from block assignments
    unique_blocks = np.unique(assignments)
    folds = []
    for block in unique_blocks:
        test_idx = np.where(assignments == block)[0]
        train_idx = np.where(assignments != block)[0]
        folds.append((train_idx, test_idx))
    
    return folds


def checkerboard_partition(
    coords: NDArray[np.float64],
    aggregation_factor: int = 2
) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Checkerboard spatial partitioning.
    
    Creates a checkerboard pattern across geographic space,
    alternating assignment between two groups.
    
    Parameters
    ----------
    coords : ndarray of shape (n_samples, 2)
        Longitude, latitude coordinates
    aggregation_factor : int
        Size of checkerboard squares relative to data extent
        
    Returns
    -------
    List of (train_indices, test_indices) tuples (2 folds)
    
    Notes
    -----
    Returns exactly 2 folds in checkerboard pattern.
    Good for strongly clustered data.
    """
    # Normalize coordinates to [0, 1]
    coords_norm = coords - coords.min(axis=0)
    coords_norm = coords_norm / coords_norm.max(axis=0)
    
    # Create checkerboard pattern
    cell_size = 1.0 / aggregation_factor
    x_cell = (coords_norm[:, 0] / cell_size).astype(int)
    y_cell = (coords_norm[:, 1] / cell_size).astype(int)
    
    # Checkerboard: (x + y) % 2
    assignments = (x_cell + y_cell) % 2
    
    fold_0_test = np.where(assignments == 0)[0]
    fold_0_train = np.where(assignments == 1)[0]
    fold_1_test = np.where(assignments == 1)[0]
    fold_1_train = np.where(assignments == 0)[0]
    
    return [(fold_0_train, fold_0_test), (fold_1_train, fold_1_test)]
