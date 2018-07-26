"""
SparsePlusLowRank unit tests
# TODO: test SVD
"""

from splr import LowRank, SparsePlusLowRank

import pytest
import scipy.sparse
import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

x = SparsePlusLowRank(scipy.sparse.rand(n, m), LowRank(np.random.random((n, p)), np.random.random((m, p))))
