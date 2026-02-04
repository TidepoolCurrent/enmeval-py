"""
enmeval-py: Automated tuning and evaluation of ecological niche models.

A Python port of ENMeval (R), designed for species distribution modeling.
"""

__version__ = "0.1.0"

from .partitioning import (
    random_kfold,
    leave_one_out,
    block_partition,
    checkerboard_partition,
)
from .evaluation import (
    calc_auc,
    calc_cbi,
    calc_omission_rate,
)
