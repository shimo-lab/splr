"""
LowRank unit tests
# TODO: run tests for several random seeds
# TODO: test initialization
"""

from splr import LowRank

import pytest
import scipy.sparse
import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

a = np.random.random((n, p))
b = np.random.random((m, p))
x = LowRank(a, b)

def test_left_mul_ndarray():
    # test valid left multiplication by 2D ndarray
    y1 = np.random.random((np.random.randint(1, 10), n))
    z1 = y1 @ x
    assert(np.allclose(z1.toarray(), y1.dot(x.toarray())))

    # test invalid left multiplication by 2D ndarray raises an error
    y2 = np.random.random((np.random.randint(1, 10), n + np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        y2 @ x

    # test left multiplication by 1D ndarray raises an error
    y3 = np.random.random(n)
    with pytest.raises(NotImplementedError):
        y3 @ x

def test_left_mul_low_rank():
    # test valid left multiplication by LowRank
    y1_nrows = np.random.randint(1, 10)
    y1_ncol = n
    y1_factor_ncol = np.random.randint(1, 10)
    y1_a = np.random.random((y1_nrows, y1_factor_ncol))
    y1_b = np.random.random((y1_ncol, y1_factor_ncol))
    y1 = LowRank(y1_a, y1_b)
    z1 = y1 @ x
    assert(np.allclose(z1.toarray(), y1.toarray().dot(x.toarray())))

    # test invalid left multiplication by LowRank raises an error
    y2_nrows = np.random.randint(1, 10)
    y2_ncol = n + np.random.randint(1, 10)
    y2_factor_ncol = np.random.randint(1, 10)
    y2_a = np.random.random((y2_nrows, y2_factor_ncol))
    y2_b = np.random.random((y2_ncol, y2_factor_ncol))
    y2 = LowRank(y2_a, y2_b)
    with pytest.raises(ValueError):
        y2 @ x

def test_left_mul_sparse():
    # test valid left multiplication by spmatrix
    y1 = scipy.sparse.rand(np.random.randint(1, 10), n)
    z1 = y1 @ x
    assert(np.allclose(z1.toarray(), y1.toarray().dot(x.toarray())))

    # test invalid left multiplication by spmatrix raises an error
    y2 = scipy.sparse.rand(np.random.randint(1, 10), n + np.random.randint(1, 10))
    with pytest.raises(ValueError):
        y2 @ x

def test_right_mul_ndarray():
    # test valid right multiplication by 2D ndarray
    y1 = np.random.random((m, np.random.randint(1, 10)))
    z1 = x @ y1
    assert(np.allclose(z1.toarray(), x.toarray().dot(y1)))

    # test invalid right multiplication by 2D ndarray raises an error
    y2 = np.random.random((m + np.random.randint(1, 10), np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        x @ y2

    # test right multiplication by 1D ndarray raises an error
    y3 = np.random.random(m)
    with pytest.raises(NotImplementedError):
        x @ y3
        
def test_right_mul_low_rank():
    # test valid right multiplication by LowRank
    y1_nrows = m
    y1_ncol = np.random.randint(1, 10)
    y1_factor_ncol = np.random.randint(1, 10)
    y1_a = np.random.random((y1_nrows, y1_factor_ncol))
    y1_b = np.random.random((y1_ncol, y1_factor_ncol))
    y1 = LowRank(y1_a, y1_b)
    z1 = x @ y1
    assert(np.allclose(z1.toarray(), x.toarray().dot(y1.toarray())))

    # test invalid right multiplication by LowRank raises an error
    y2_nrows = m + np.random.randint(1, 10)
    y2_ncol = np.random.randint(1, 10)
    y2_factor_ncol = np.random.randint(1, 10)
    y2_a = np.random.random((y2_nrows, y2_factor_ncol))
    y2_b = np.random.random((y2_ncol, y2_factor_ncol))
    y2 = LowRank(y2_a, y2_b)
    with pytest.raises(ValueError):
        x @ y2
        
def test_right_mul_sparse():
    # test valid right multiplication by spmatrix
    y1 = scipy.sparse.rand(m, np.random.randint(1, 10))
    z1 = x @ y1
    assert(np.allclose(z1.toarray(), x.toarray().dot(y1.toarray())))

    # test invalid right multiplication by spmatrix raises an error
    y2 = scipy.sparse.rand(m + np.random.randint(1, 10), np.random.randint(1, 10))
    with pytest.raises(ValueError):
        x @ y2
