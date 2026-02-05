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
    
    def test_cbi_r_parity(self):
        """Compare CBI to R ecospat.boyce() with hardcoded data."""
        from enmeval.boyce import calc_boyce
        
        # Hardcoded test data (same as R test)
        fit = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                        0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05])
        obs = np.array([0.7, 0.8, 0.9, 0.75, 0.85])
        
        # Python results
        py_cbi_dup, _ = calc_boyce(fit, obs, res=100, rm_duplicate=True)
        py_cbi_nodup, _ = calc_boyce(fit, obs, res=100, rm_duplicate=False)
        
        # R reference values (computed with ecospat.boyce)
        # R CBI (rm.dup=TRUE): 0.632
        # R CBI (rm.dup=FALSE): 0.813
        r_cbi_dup = 0.632
        r_cbi_nodup = 0.813
        
        assert abs(py_cbi_dup - r_cbi_dup) < 0.01, \
            f"Python CBI (rm_dup) {py_cbi_dup:.3f} != R {r_cbi_dup}"
        assert abs(py_cbi_nodup - r_cbi_nodup) < 0.01, \
            f"Python CBI (no rm_dup) {py_cbi_nodup:.3f} != R {r_cbi_nodup}"


if __name__ == "__main__":
    print(f"R available: {r_available()}")
    if r_available():
        print("Running quick test...")
        t = TestAUCParity()
        t.test_basic_auc()
        print("AUC parity: PASS")
