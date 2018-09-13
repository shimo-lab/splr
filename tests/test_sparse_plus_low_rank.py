"""
SparsePlusLowRank unit tests
# TODO: run tests for several random seeds
# TODO: test SVD
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splr import LowRank, SparsePlusLowRank

import pytest
import scipy.sparse
import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

sparse = scipy.sparse.rand(n, m)

a = np.random.random((n, p))
b = np.random.random((m, p))
low_rank = LowRank(a, b)

x = SparsePlusLowRank(sparse, low_rank)

def test_left_mul_ndarray():
    # test valid left multiplication by 2D ndarray
    y1 = np.random.random((np.random.randint(1, 10), n))
    z1 = y1 @ x
    assert(np.allclose(z1, y1.dot(x.toarray())))

    # test invalid left multiplication by 2D ndarray raises an error
    y2 = np.random.random((np.random.randint(1, 10), n + np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        y2 @ x

    # test left multiplication by 1D ndarray raises an error
    y3 = np.random.random(n)
    with pytest.raises(NotImplementedError):
        y3 @ x

def test_right_mul_ndarray():
    # test valid right multiplication by 2D ndarray
    y1 = np.random.random((m, np.random.randint(1, 10)))
    z1 = x @ y1
    assert(np.allclose(z1, x.toarray().dot(y1)))

    # test invalid right multiplication by 2D ndarray raises an error
    y2 = np.random.random((m + np.random.randint(1, 10), np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        x @ y2

    # test right multiplication by 1D ndarray raises an error
    y3 = np.random.random(m)
    with pytest.raises(NotImplementedError):
        x @ y3