# R Parity Validation Plan

## Goal
Reproduce ENMeval R outputs with enmeval-py using identical data.

## Data Source
- `bvariegatus` - 476 occurrence records (Brown Throated Sloth)
- Built into ENMeval R package
- Downloaded from GBIF

## Validation Process

### Step 1: Generate R Reference Data
```r
library(ENMeval)
data(bvariegatus)

# Run with specific settings
results <- ENMevaluate(
  occs = bvariegatus,
  envs = <worldclim data>,
  partitions = "randomkfold",
  tune.args = list(rm = c(0.5, 1, 2), fc = c("L", "LQ")),
  n.folds = 5
)

# Save results to compare
write.csv(results@results, "r_reference_results.csv")
```

### Step 2: Run Python with Same Data
```python
from enmeval import enmeval

results = enmeval(
    presence_coords=bvariegatus_coords,
    presence_envs=bvariegatus_envs,
    background_envs=background_envs,
    model_fn=fit_maxent,
    regularization_multipliers=[0.5, 1, 2],
    feature_classes=['L', 'LQ'],
    n_folds=5,
    random_state=48,  # match R's set.seed(48)
)
```

### Step 3: Compare
- AUC values within ±0.01
- CBI values within ±0.05
- Omission rates within ±0.02
- Same model selected as "best"

## Status
- [ ] Install R + ENMeval in WSL
- [ ] Generate reference data
- [ ] Add Python comparison test
- [ ] Document results in README
