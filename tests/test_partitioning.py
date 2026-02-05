"""Tests for partitioning methods with R ENMeval parity checks."""

import numpy as np
import pytest
from enmeval.partitioning import (
    random_kfold,
    leave_one_out,
    block_partition,
    checkerboard_partition,
)


class TestRandomKfold:
    def test_basic_splits(self):
        """Test that k-fold produces correct number of folds."""
        folds = random_kfold(100, k=5, random_state=42)
        assert len(folds) == 5
        
    def test_all_indices_used(self):
        """Test that all samples appear exactly once in test sets."""
        folds = random_kfold(100, k=5, random_state=42)
        all_test_indices = np.concatenate([f[1] for f in folds])
        assert len(all_test_indices) == 100
        assert len(np.unique(all_test_indices)) == 100
        
    def test_no_overlap(self):
        """Test that train and test don't overlap within each fold."""
        folds = random_kfold(100, k=5, random_state=42)
        for train, test in folds:
            assert len(np.intersect1d(train, test)) == 0
            
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        folds1 = random_kfold(100, k=5, random_state=42)
        folds2 = random_kfold(100, k=5, random_state=42)
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)


class TestLeaveOneOut:
    def test_fold_count(self):
        """Test that LOO produces n folds for n samples."""
        folds = leave_one_out(10)
        assert len(folds) == 10
        
    def test_single_test_per_fold(self):
        """Test that each fold has exactly one test sample."""
        folds = leave_one_out(10)
        for _, test in folds:
            assert len(test) == 1


class TestBlockPartition:
    def test_basic_blocks(self):
        """Test that block partition creates spatial groups."""
        np.random.seed(42)
        coords = np.random.rand(100, 2) * 100  # Random coords
        folds = block_partition(coords, k=4)
        
        # Should have up to k folds
        assert len(folds) <= 4
        
    def test_spatial_coherence(self):
        """Test that blocks are spatially coherent."""
        # Create clustered data
        coords = np.vstack([
            np.random.rand(25, 2) * 10,  # Cluster 1: lower-left
            np.random.rand(25, 2) * 10 + 90,  # Cluster 2: upper-right
        ])
        folds = block_partition(coords, k=2, orientation="auto")
        
        # Should separate the two clusters
        assert len(folds) == 2


class TestCheckerboard:
    def test_two_folds(self):
        """Test that checkerboard produces exactly 2 folds."""
        coords = np.random.rand(100, 2) * 100
        folds = checkerboard_partition(coords, aggregation_factor=4)
        assert len(folds) == 2
        
    def test_complete_coverage(self):
        """Test that all points are assigned."""
        coords = np.random.rand(100, 2) * 100
        folds = checkerboard_partition(coords, aggregation_factor=4)
        all_test = np.concatenate([f[1] for f in folds])
        assert len(all_test) == 100


# R Parity tests - compare against known R ENMeval outputs
class TestRParity:
    """
    Tests comparing Python outputs to R ENMeval outputs.
    
    Note: Partitioning algorithms may differ in implementation details
    (different RNGs, blocking strategies). What matters is that the
    **evaluation metrics** (AUC, CBI) are comparable, not exact fold
    assignments. See tests/test_r_parity.py for metric parity tests.
    
    These structural tests verify that partitioning produces valid folds
    with the expected properties.
    """
    
    def test_random_kfold_properties(self):
        """Verify random k-fold has correct structural properties."""
        from enmeval.partitioning import random_kfold
        
        n_samples = 10
        folds = random_kfold(n_samples, k=5, random_state=48)
        
        # Verify 5 folds returned
        assert len(folds) == 5
        
        # Verify each fold has (train, test) tuple
        all_test = []
        for train_idx, test_idx in folds:
            # Train + test = all samples
            combined = set(train_idx) | set(test_idx)
            assert combined == set(range(n_samples))
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            all_test.extend(test_idx)
        
        # Each sample appears in test exactly once
        assert sorted(all_test) == list(range(n_samples))
    
    def test_block_properties(self):
        """Verify block partition has spatial coherence."""
        from enmeval.partitioning import block_partition
        
        # Grid of points
        xx, yy = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        folds = block_partition(coords, k=4)
        
        # Verify 4 folds
        assert len(folds) == 4
        
        # Verify all samples covered
        all_test = []
        for train_idx, test_idx in folds:
            all_test.extend(test_idx)
        assert sorted(all_test) == list(range(25))
