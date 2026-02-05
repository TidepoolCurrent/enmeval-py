# enmeval-py

Python implementation of [ENMeval](https://github.com/jamiemkass/ENMeval) - automated tuning and evaluation of ecological niche models.

## Status: ⚠️ Pre-alpha - Functional but NOT YET VALIDATED against R implementation

This is a Python port of the R package ENMeval, designed to work with [elapid](https://github.com/earth-chris/elapid) for MaxEnt modeling.

## Testing Status

### What's been tested:
- ✅ **Integration**: Code runs with elapid MaxEnt backend
- ✅ **Smoke tests**: AUC behaves as expected on synthetic data (see `examples/validate_elapid_integration.py`)

### What has NOT been tested:
- ❌ **R Parity**: No comparison to R ENMeval outputs on same data
- ❌ **Reference data**: Not tested with bvariegatus or other published datasets
- ❌ **Numerical verification**: Results may differ from R ENMeval (not yet quantified)

### ⚠️ Use with caution
Do not use for production research without independent validation against R ENMeval.

### Validation Roadmap
- [ ] Test with bvariegatus reference data (R ENMeval example)
- [ ] Compare outputs to R ENMeval (AUC, CBI, model selection)
- [ ] Document numerical differences and acceptable tolerances
- [ ] Achieve parity or document why differences exist

## Features

### Partitioning Methods
- [x] Random k-fold
- [x] Leave-one-out (jackknife)
- [x] Block (spatial)
- [x] Checkerboard (spatial)

### Evaluation Metrics
- [x] AUC (Area Under ROC Curve)
- [x] Continuous Boyce Index (CBI)
- [x] Omission rates (at various thresholds)
- [x] AICc

### Tuning
- [x] Grid search across regularization multipliers
- [x] Feature class combinations
- [x] Cross-validation orchestration
- [x] Best model selection (by AUC or AICc)

## Quick Start

```python
from enmeval import enmeval
from elapid import MaxentModel
import numpy as np

# Your data
coords = ...  # (n_presence, 2) lon/lat
pres_env = ...  # (n_presence, n_features)
bg_env = ...  # (n_background, n_features)

# Wrapper for your model
def fit_maxent(train_pres, train_bg, rm, fc):
    X = np.vstack([train_pres, train_bg])
    y = np.array([1]*len(train_pres) + [0]*len(train_bg))
    model = MaxentModel(tau=rm)
    model.fit(X, y)
    return model

# Run tuning
results = enmeval(
    presence_coords=coords,
    presence_envs=pres_env,
    background_envs=bg_env,
    model_fn=fit_maxent,
    regularization_multipliers=[0.5, 1, 2, 4],
    feature_classes=['L', 'LQ', 'LQH'],
    n_folds=5,
)

print(f"Best AUC: {results.best_auc.auc_test:.3f}")
print(f"Best config: rm={results.best_auc.regularization}, fc={results.best_auc.feature_classes}")
```

## Test Suite

```bash
# 23 tests, all passing
pytest tests/ -v

# Run validation example
python examples/validate_elapid_integration.py
```

## Installation

```bash
# From source (recommended for now)
git clone https://github.com/TidepoolCurrent/enmeval-py
cd enmeval-py
pip install -e .

# Dependencies
pip install numpy scipy elapid
```

## Why?

ENMeval is essential for rigorous species distribution modeling, but only exists in R. This port enables:
- Integration with Python ML pipelines
- Use with modern geospatial tools (geopandas, rasterio)
- Deployment in production environments

## Roadmap

- [ ] R parity tests (compare outputs to R ENMeval)
- [ ] Real species data examples
- [ ] PyPI publication
- [ ] Full documentation

## References

- Kass et al. (2021). ENMeval 2.0: redesigned for customizable and reproducible modeling of species' niches and distributions. Methods in Ecology and Evolution, 12: 1602-1608.
- Original R package: https://github.com/jamiemkass/ENMeval

## License

GPL-3.0 (same as original R package)
