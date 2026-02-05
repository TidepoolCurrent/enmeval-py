# enmeval-py - Agent Quick Reference

*What this tool does, how to use it, what can go wrong.*

## What It Does

**Input:** Species occurrence coordinates + environmental data + background points
**Output:** Ranked MaxEnt models with evaluation metrics (AUC, CBI, omission rate, AICc)

**Purpose:** Find the best regularization/feature combination for your species distribution model.

## When To Use

You have:
- Presence points (where species was observed)
- Environmental layers (climate, terrain, etc.)
- Background points (random sample of available environment)

You want:
- Optimal model settings (not default MaxEnt)
- Cross-validated performance metrics
- Defensible model selection

## Dependencies

```
numpy scipy elapid
```

**elapid** provides the MaxEnt backend. Must be installed: `pip install elapid`

## Minimal Working Example

```python
import numpy as np
from enmeval import enmeval
from elapid import MaxentModel

# 1. Prepare data
presence_coords = np.array([[lon1, lat1], [lon2, lat2], ...])  # (n, 2)
presence_envs = np.array([[temp1, precip1], [temp2, precip2], ...])  # (n, features)
background_envs = np.array([[...], [...], ...])  # (m, features) - same features!

# 2. Define model function
def fit_maxent(train_pres, train_bg, rm, fc):
    """rm = regularization multiplier, fc = feature classes (ignored by elapid)"""
    X = np.vstack([train_pres, train_bg])
    y = np.array([1]*len(train_pres) + [0]*len(train_bg))
    model = MaxentModel(tau=rm)
    model.fit(X, y)
    return model

# 3. Run tuning
results = enmeval(
    presence_coords=presence_coords,
    presence_envs=presence_envs,
    background_envs=background_envs,
    model_fn=fit_maxent,
    regularization_multipliers=[0.5, 1, 2, 4],
    feature_classes=['L', 'LQ', 'LQH'],
    n_folds=5,
)

# 4. Get best model
print(f"Best AUC: {results.best_auc.auc_test:.3f}")
print(f"Config: rm={results.best_auc.regularization}")
```

## Inputs (Detailed)

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `presence_coords` | ndarray (n, 2) | Lon/lat of occurrences | Yes |
| `presence_envs` | ndarray (n, k) | Environmental values at presence points | Yes |
| `background_envs` | ndarray (m, k) | Environmental values at background points | Yes |
| `model_fn` | callable | Function that returns fitted model | Yes |
| `regularization_multipliers` | list[float] | Values to test | Yes |
| `feature_classes` | list[str] | Feature combos to test | Yes |
| `n_folds` | int | Cross-validation folds (default: 5) | No |
| `partition_method` | str | 'random', 'loo', 'block', 'checkerboard' | No |

## Outputs

`results` is an `ENMevalResults` object:

```python
results.all_results      # List of all model results
results.best_auc         # Best model by test AUC
results.best_aicc        # Best model by AICc (if calculable)
results.summary()        # DataFrame of all results
```

Each result has:
- `.auc_train`, `.auc_test` - AUC scores
- `.cbi` - Continuous Boyce Index
- `.omission_rate` - At 10th percentile threshold
- `.regularization`, `.feature_classes` - Settings used

## What Can Go Wrong

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: X and y length mismatch` | presence_envs rows ≠ presence_coords rows | Check array shapes |
| `Model failed to converge` | Too few points or bad regularization | Increase rm, check data quality |
| `All AUC = 0.5` | No signal in data | Check environmental layers match species |
| `elapid not found` | Missing dependency | `pip install elapid` |

## Verification

After running, verify:

1. **AUC should vary:** If all models have same AUC, something's wrong
2. **Best AUC > 0.7:** Lower suggests weak model or bad data
3. **Test AUC ≤ Train AUC:** If test > train, likely data leakage
4. **CBI > 0:** Negative CBI suggests model is worse than random

## Comparison to R ENMeval

| Feature | R ENMeval | enmeval-py |
|---------|-----------|------------|
| MaxEnt backend | maxent.jar or maxnet | elapid |
| Partitioning | block, checkerboard, random, user | ✓ same |
| Metrics | AUC, CBI, OR, AICc | ✓ same |
| R parity tested? | — | ⚠️ NOT YET |

**Warning:** Results may differ from R ENMeval. Use for exploration, verify important results against R.

## Links

- Repo: https://github.com/TidepoolCurrent/enmeval-py
- elapid: https://github.com/earth-chris/elapid
- R ENMeval: https://github.com/jamiemkass/ENMeval
