"""
SparsePlusLowRank unit tests
"""

from splr import LowRank, SparsePlusLowRank

import pytest
import scipy.sparse
import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

x = SparsePlusLowRank(scipy.sparse.rand(n, m), LowRank(np.random.random((n, p)), np.random.random((m, p))))

def test_left_mul():
    # test valid left multiplication by matrix
    y1 = np.random.random((np.random.randint(1, 10), n))
    z1 = x.left_mul(y1)
    assert(np.allclose(z1, y1 * x.sparse + x.low_rank.left_mul(y1)))

    # test invalid left multiplication by matrix raises an error
    y2 = np.random.random((np.random.randint(1, 10), n + np.random.randint(1, 10)))
    with pytest.raises(AssertionError):
        x.left_mul(y2)

    # test left multiplication by vector raises an error
    y3 = np.random.random(n)
    with pytest.raises(AssertionError):
        x.left_mul(y3)

def test_right_mul():
    # test valid right multiplication by matrix
    y1 = np.random.random((m, np.random.randint(1, 10)))
    z1 = x.right_mul(y1)
    assert(np.allclose(z1, x.sparse.dot(y1) + x.low_rank.right_mul(y1)))

    # test invalid right multiplication by matrix raises an error
    y2 = np.random.random((m + np.random.randint(1, 10), np.random.randint(1, 10)))
    with pytest.raises(AssertionError):
        x.right_mul(y2)

    # test right multiplication by vector raises an error
    y3 = np.random.random(m)
    with pytest.raises(AssertionError):
        x.right_mul(y3)

# TODO: test SVD
