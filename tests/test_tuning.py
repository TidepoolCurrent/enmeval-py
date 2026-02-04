"""Tests for model tuning workflow."""

import numpy as np
import pytest
from enmeval.tuning import enmeval, TuningResult, ENMevalResults


class MockModel:
    """Mock model for testing tuning workflow."""
    
    def __init__(self, rm, fc):
        self.rm = rm
        self.fc = fc
        
    def predict(self, X):
        # Return mock predictions based on first feature
        # Higher rm = smoother predictions (less overfitting)
        noise = np.random.rand(len(X)) * (1 / self.rm)
        return np.clip(X[:, 0] / X[:, 0].max() + noise, 0, 1)


def mock_model_fn(train_pres, train_bg, rm, fc):
    """Mock model fitting function."""
    return MockModel(rm, fc)


class TestTuningWorkflow:
    def test_basic_tuning(self):
        """Test that tuning runs without error."""
        np.random.seed(42)
        n_pres = 50
        n_bg = 200
        n_features = 5
        
        coords = np.random.rand(n_pres, 2) * 100
        pres_env = np.random.rand(n_pres, n_features)
        bg_env = np.random.rand(n_bg, n_features)
        
        results = enmeval(
            presence_coords=coords,
            presence_envs=pres_env,
            background_envs=bg_env,
            model_fn=mock_model_fn,
            regularization_multipliers=[1, 2],
            feature_classes=['L', 'LQ'],
            n_folds=3,
            random_state=42,
        )
        
        # Should have 2x2 = 4 results
        assert len(results.results) == 4
        
    def test_best_model_selection(self):
        """Test that best model is correctly identified."""
        np.random.seed(42)
        coords = np.random.rand(50, 2) * 100
        pres_env = np.random.rand(50, 5)
        bg_env = np.random.rand(200, 5)
        
        results = enmeval(
            presence_coords=coords,
            presence_envs=pres_env,
            background_envs=bg_env,
            model_fn=mock_model_fn,
            regularization_multipliers=[1, 2, 4],
            feature_classes=['L'],
            n_folds=3,
            random_state=42,
        )
        
        # Best model should have highest test AUC
        all_aucs = [r.auc_test for r in results.results]
        assert results.best_auc.auc_test == max(all_aucs)
        
    def test_partition_methods(self):
        """Test different partitioning methods work."""
        np.random.seed(42)
        coords = np.random.rand(50, 2) * 100
        pres_env = np.random.rand(50, 5)
        bg_env = np.random.rand(200, 5)
        
        for method in ['randomkfold', 'block', 'checkerboard']:
            results = enmeval(
                presence_coords=coords,
                presence_envs=pres_env,
                background_envs=bg_env,
                model_fn=mock_model_fn,
                regularization_multipliers=[1],
                feature_classes=['L'],
                partitions=method,
                n_folds=4,
                random_state=42,
            )
            assert len(results.results) == 1


class TestTuningResult:
    def test_auc_diff(self):
        """Test AUC difference calculation."""
        result = TuningResult(
            regularization=1.0,
            feature_classes='L',
            auc_train=0.9,
            auc_test=0.8,
            cbi=0.5,
            omission_rate=0.1,
            aicc=None,
            n_params=5,
        )
        assert abs(result.auc_diff - 0.1) < 1e-9
