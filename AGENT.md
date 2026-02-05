# enmeval-py - Agent Quick Reference

*What this tool does, how to use it, what can go wrong.*

## Installation

```bash
# Clone the repo (not on PyPI yet)
git clone https://github.com/TidepoolCurrent/enmeval-py
cd enmeval-py
pip install -e .

# Dependencies
pip install numpy scipy elapid rasterio
```

## What It Does

**Input:** Species occurrence coordinates + environmental raster values + background sample
**Output:** Ranked MaxEnt models with evaluation metrics, best model for predictions

**Purpose:** Tune MaxEnt regularization to avoid overfitting. Default MaxEnt often overfits.

## Jargon Definitions

| Term | Meaning |
|------|---------|
| **MaxEnt** | Maximum Entropy model - predicts habitat suitability from presence-only data |
| **AUC** | Area Under ROC Curve (0-1). Higher = better discrimination. >0.7 decent, >0.8 good |
| **CBI** | Continuous Boyce Index (-1 to 1). Measures if predictions match presence density. >0 good |
| **Regularization** | Penalty on model complexity. Higher = simpler model. Prevents overfitting |
| **Feature classes** | L=Linear, Q=Quadratic, H=Hinge, P=Product, T=Threshold. Complexity of response curves |

## When To Use

**Use this when:**
- You have GPS coordinates where a species was observed
- You have environmental rasters (climate, terrain, etc.)
- You want to predict where else the species could live
- You need to justify model settings to reviewers (not just "I used defaults")

**Don't use when:**
- You have presence AND absence data (use other methods)
- You have <20 occurrence points (too few for cross-validation)

## Full Working Example

```python
import numpy as np
import rasterio
from enmeval import enmeval
from elapid import MaxentModel

# ============================================
# STEP 1: Load occurrence coordinates
# ============================================
# Format: (longitude, latitude) - THIS ORDER MATTERS
# Longitude first (x), latitude second (y)
presence_coords = np.array([
    [-122.419, 37.775],  # San Francisco
    [-122.272, 37.871],  # Berkeley
    [-121.886, 37.338],  # San Jose
    [-122.031, 36.974],  # Santa Cruz
])  # Shape: (n_occurrences, 2)

# ============================================
# STEP 2: Extract environmental values at coordinates
# ============================================
def extract_raster_values(coords, raster_paths):
    """Extract values from multiple rasters at given coordinates.
    
    Args:
        coords: array of (lon, lat) pairs
        raster_paths: list of paths to .tif files
    Returns:
        array of shape (n_coords, n_rasters)
    """
    values = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            # rasterio.sample wants [(lon, lat), ...] - same order we use
            samples = list(src.sample(coords))
            values.append([s[0] for s in samples])
    return np.array(values).T  # Shape: (n_coords, n_rasters)

# Your environmental layers (example paths)
env_rasters = [
    'data/bio1_temp.tif',    # Mean annual temperature
    'data/bio12_precip.tif', # Annual precipitation
    'data/elevation.tif',    # Elevation
]

presence_envs = extract_raster_values(presence_coords, env_rasters)
# Shape: (4, 3) - 4 occurrences, 3 environmental variables

# ============================================
# STEP 3: Generate background points
# ============================================
def generate_background(raster_path, n_points=1000, seed=42):
    """Generate random background points within raster extent."""
    np.random.seed(seed)
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        # Random points in bounding box
        lons = np.random.uniform(bounds.left, bounds.right, n_points)
        lats = np.random.uniform(bounds.bottom, bounds.top, n_points)
        coords = np.column_stack([lons, lats])
        
        # Filter to points with valid data (not nodata)
        samples = list(src.sample(coords))
        valid = [i for i, s in enumerate(samples) if s[0] != src.nodata]
        return coords[valid]

bg_coords = generate_background(env_rasters[0], n_points=1000)
background_envs = extract_raster_values(bg_coords, env_rasters)
# Shape: (~1000, 3) - background points, same 3 variables

# ============================================
# STEP 4: Define model fitting function
# ============================================
def fit_maxent(train_pres, train_bg, rm, fc):
    """
    Args:
        train_pres: environmental values at training presence points
        train_bg: environmental values at background points
        rm: regularization multiplier (higher = simpler model)
        fc: feature classes (ignored by elapid, kept for R compatibility)
    Returns:
        fitted model with .predict() method
    """
    X = np.vstack([train_pres, train_bg])
    y = np.array([1]*len(train_pres) + [0]*len(train_bg))
    model = MaxentModel(beta_multiplier=rm)  # beta_multiplier = regularization
    model.fit(X, y)
    return model

# ============================================
# STEP 5: Run tuning
# ============================================
results = enmeval(
    presence_coords=presence_coords,
    presence_envs=presence_envs,
    background_envs=background_envs,
    model_fn=fit_maxent,
    regularization_multipliers=[0.5, 1.0, 2.0, 4.0],
    feature_classes=['L'],  # Ignored by elapid but required
    n_folds=4,  # 4-fold cross-validation (need n_folds <= n_occurrences)
)

# ============================================
# STEP 6: Get best model and make predictions
# ============================================
print(f"Best test AUC: {results.best_auc.auc_test:.3f}")
print(f"Best regularization: {results.best_auc.regularization}")

# The best model object (for predictions)
best_model = results.best_auc.model

# Predict on new environmental data
new_env = np.array([[15.0, 800, 100]])  # temp, precip, elevation
prediction = best_model.predict(new_env)
print(f"Habitat suitability: {prediction[0]:.3f}")  # 0-1 scale
```

## Return Value Structure

```python
results.all_results      # List[ModelResult] - all tested combinations
results.best_auc         # ModelResult - best by test AUC
results.best_aicc        # ModelResult or None - best by AICc
results.summary()        # pandas DataFrame of all results

# Each ModelResult has:
result.auc_train         # float - training AUC
result.auc_test          # float - test AUC (cross-validated)
result.cbi               # float - Continuous Boyce Index
result.omission_rate     # float - at 10th percentile threshold
result.regularization    # float - the rm value used
result.feature_classes   # str - the fc value used
result.model             # fitted model object with .predict()
```

## What Can Go Wrong

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: X and y length mismatch` | presence_envs rows ≠ presence_coords rows | Check array shapes match |
| `Model failed to converge` | Too few points or extreme regularization | Try rm=1.0, check for NaN in data |
| `All AUC = 0.5` | No environmental signal | Verify rasters cover occurrence area |
| `elapid not found` | Missing dependency | `pip install elapid` |
| `rasterio not found` | Missing for data prep | `pip install rasterio` |
| Silent wrong results | Lat/lon order swapped | Verify: longitude FIRST, latitude SECOND |

## Verification Checklist

After running, check:

1. **AUC varies across models** — If all identical, something's wrong
2. **Test AUC ≤ Train AUC** — If test > train, likely data leakage
3. **Best AUC > 0.7** — Lower is okay for cryptic species, but investigate
4. **CBI > 0** — Negative means model is worse than random
5. **Predictions make biological sense** — Check known locations get high scores

## Comparison to R ENMeval

| Feature | R ENMeval | enmeval-py |
|---------|-----------|------------|
| MaxEnt backend | maxent.jar or maxnet | elapid |
| Partitioning | block, checkerboard, random | ✓ same |
| Metrics | AUC, CBI, OR, AICc | ✓ same |
| **R parity tested?** | — | ⚠️ **NOT YET** |

**Warning:** Results may differ from R ENMeval. Validate important analyses against R.

## Links

- **This repo:** https://github.com/TidepoolCurrent/enmeval-py
- **elapid (MaxEnt):** https://github.com/earth-chris/elapid
- **R ENMeval:** https://github.com/jamiemkass/ENMeval
- **rasterio docs:** https://rasterio.readthedocs.io/
