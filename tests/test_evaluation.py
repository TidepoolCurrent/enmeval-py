"""Tests for evaluation metrics with R ENMeval parity checks."""

import numpy as np
import pytest
from enmeval.evaluation import calc_auc, calc_cbi, calc_omission_rate, calc_aicc


class TestAUC:
    def test_perfect_discrimination(self):
        """Test AUC = 1 when presence always > background."""
        presence = np.array([0.9, 0.8, 0.95, 0.85])
        background = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
        auc = calc_auc(presence, background)
        assert auc == 1.0
        
    def test_random_discrimination(self):
        """Test AUC ≈ 0.5 when predictions are random."""
        np.random.seed(42)
        presence = np.random.rand(1000)
        background = np.random.rand(1000)
        auc = calc_auc(presence, background)
        assert 0.45 < auc < 0.55
        
    def test_inverse_discrimination(self):
        """Test AUC = 0 when presence always < background."""
        presence = np.array([0.1, 0.2, 0.15])
        background = np.array([0.9, 0.8, 0.95, 0.85])
        auc = calc_auc(presence, background)
        assert auc == 0.0


class TestCBI:
    def test_perfect_calibration(self):
        """Test CBI ≈ 1 when presences increase with predictions."""
        # Presences concentrated at high predictions
        presence = np.random.uniform(0.7, 1.0, 100)
        # Background uniform
        background = np.random.uniform(0.0, 1.0, 1000)
        cbi = calc_cbi(presence, background)
        assert cbi > 0.5
        
    def test_random_calibration(self):
        """Test CBI ≈ 0 when predictions are uninformative."""
        np.random.seed(42)
        presence = np.random.rand(100)
        background = np.random.rand(1000)
        cbi = calc_cbi(presence, background)
        assert -0.5 < cbi < 0.5  # Wider tolerance for random data


class TestOmissionRate:
    def test_zero_omission(self):
        """Test 0% omission when all presences above threshold."""
        presence = np.array([0.5, 0.6, 0.7, 0.8])
        rate, thresh = calc_omission_rate(presence, threshold=0.4)
        assert rate == 0.0
        
    def test_full_omission(self):
        """Test 100% omission when all presences below threshold."""
        presence = np.array([0.1, 0.2, 0.3])
        rate, thresh = calc_omission_rate(presence, threshold=0.5)
        assert rate == 1.0
        
    def test_percentile_threshold(self):
        """Test percentile-based threshold calculation."""
        presence = np.linspace(0, 1, 100)
        rate, thresh = calc_omission_rate(presence, percentile=10)
        # 10th percentile should give ~10% omission
        assert 0.08 < rate < 0.12


class TestAICc:
    def test_penalty_increases_with_params(self):
        """Test that more parameters increases AICc."""
        ll = -100  # Same log-likelihood
        n = 50
        aicc_simple = calc_aicc(ll, n_params=2, n_samples=n)
        aicc_complex = calc_aicc(ll, n_params=10, n_samples=n)
        assert aicc_complex > aicc_simple


# R Parity tests
class TestRParity:
    @pytest.mark.skip(reason="R reference data not yet generated")
    def test_auc_parity(self):
        """Compare AUC to R ENMeval calc.auc()."""
        pass
    
    @pytest.mark.skip(reason="R reference data not yet generated")
    def test_cbi_parity(self):
        """Compare CBI to R ENMeval calc.cbi()."""
        pass
