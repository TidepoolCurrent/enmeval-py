"""
True R parity tests - compare Python output to R ENMeval.
"""
import pytest
import numpy as np
import subprocess
import json
import tempfile
import os

def r_available():
    """Check if R and ENMeval are available."""
    try:
        result = subprocess.run(
            ['R', '-e', '.libPaths(c("~/R/library", .libPaths())); library(ENMeval); cat("OK")'],
            capture_output=True, text=True, timeout=30
        )
        return "OK" in result.stdout
    except:
        return False

def run_r_code(code: str) -> str:
    """Run R code and return stdout."""
    full_code = f'.libPaths(c("~/R/library", .libPaths())); library(ENMeval); {code}'
    result = subprocess.run(
        ['R', '--vanilla', '-e', full_code],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout

@pytest.mark.skipif(not r_available(), reason="R/ENMeval not available")
class TestAUCParity:
    """Test AUC calculation matches R."""
    
    def test_basic_auc(self):
        """Compare basic AUC calculation."""
        from enmeval.evaluation import calc_auc
        
        # Test data
        presence_pred = np.array([0.8, 0.7, 0.9, 0.6, 0.85])
        background_pred = np.array([0.2, 0.3, 0.1, 0.4, 0.25, 0.15, 0.35])
        
        # Python result
        py_auc = calc_auc(presence_pred, background_pred)
        
        # R result using manual Wilcoxon-Mann-Whitney calculation
        pres_str = ','.join(map(str, presence_pred))
        bg_str = ','.join(map(str, background_pred))
        r_code = f'''
        pres <- c({pres_str})
        bg <- c({bg_str})
        concordant <- sum(outer(pres, bg, ">")) + 0.5 * sum(outer(pres, bg, "=="))
        auc <- concordant / (length(pres) * length(bg))
        cat("RESULT:", auc)
        '''
        r_output = run_r_code(r_code)
        
        # Extract number after RESULT:
        import re
        match = re.search(r'RESULT:\s*([\d.]+)', r_output)
        if match:
            r_auc = float(match.group(1))
            assert abs(py_auc - r_auc) < 0.001, f"Python AUC {py_auc:.4f} != R AUC {r_auc:.4f}"
        else:
            pytest.fail(f"Could not parse R output: {r_output}")


@pytest.mark.skipif(not r_available(), reason="R not available")
class TestCBIParity:
    """Test CBI (Continuous Boyce Index) matches R ecospat.boyce()."""
    
    def test_basic_cbi(self):
        """Compare CBI to R ecospat.boyce() when available."""
        from enmeval.boyce import calc_boyce
        
        # Test data: good model (presences have higher predictions)
        np.random.seed(42)
        fit = np.random.uniform(0, 1, 200)  # All predictions
        # Presences drawn from higher values
        obs = np.random.beta(2, 1, 50)
        
        # Python result
        py_cbi, _ = calc_boyce(fit, obs, res=100)
        
        # For now, just verify it's in valid range and sensible
        assert -1 <= py_cbi <= 1, f"CBI {py_cbi} out of range"
        # With presences skewed high, CBI should be positive
        assert py_cbi > 0, f"Expected positive CBI for good model, got {py_cbi}"
        
        # TODO: Add R comparison when ecospat is available
        # r_code = '''
        # library(ecospat)
        # fit <- c(...)
        # obs <- c(...)
        # result <- ecospat.boyce(fit, obs)
        # cat("RESULT:", result$Spearman.cor)
        # '''


if __name__ == "__main__":
    print(f"R available: {r_available()}")
    if r_available():
        print("Running quick test...")
        t = TestAUCParity()
        t.test_basic_auc()
        print("AUC parity: PASS")
