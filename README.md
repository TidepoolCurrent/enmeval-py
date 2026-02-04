# enmeval-py

Python implementation of [ENMeval](https://github.com/jamiemkass/ENMeval) - automated tuning and evaluation of ecological niche models.

## Status: 🚧 In Development

This is a Python port of the R package ENMeval, designed to work with [elapid](https://github.com/earth-chris/elapid) for MaxEnt modeling.

## Why?

ENMeval is essential for rigorous species distribution modeling, but only exists in R. This port enables:
- Integration with Python ML pipelines
- Use with modern geospatial tools (geopandas, rasterio)
- Deployment in production environments

## Features (Planned)

### Partitioning Methods
- [ ] Random k-fold
- [ ] Leave-one-out (jackknife)
- [ ] Block (spatial)
- [ ] Checkerboard (spatial)
- [ ] User-defined

### Evaluation Metrics
- [ ] AUC (Area Under ROC Curve)
- [ ] Continuous Boyce Index (CBI)
- [ ] Omission rates (at various thresholds)
- [ ] AICc

### Tuning
- [ ] Grid search across regularization multipliers
- [ ] Feature class combinations
- [ ] Cross-validation orchestration

## Verification

All functions are tested against R ENMeval outputs to ensure numeric parity.

```bash
# Run verification tests
pytest tests/ -v
```

## Installation

```bash
pip install enmeval-py  # not yet published
```

## References

- Kass et al. (2021). ENMeval 2.0: redesigned for customizable and reproducible modeling of species' niches and distributions. Methods in Ecology and Evolution, 12: 1602-1608.
- Original R package: https://github.com/jamiemkass/ENMeval

## License

GPL-3.0 (same as original R package)
