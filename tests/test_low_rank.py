"""
LowRank unit tests
# TODO: run tests for several random seeds
# TODO: test initialization
"""

from splr import LowRank

import pytest
import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

a = np.random.random((n, p))
b = np.random.random((m, p))
x = LowRank(a, b)

def test_left_mul():
    # test valid left multiplication by matrix
    y1 = np.random.random((np.random.randint(1, 10), n))
    z1 = y1 @ x
    assert(np.allclose(z1.toarray(), y1.dot(x.a).dot(x.b.T)))

    # test invalid left multiplication by matrix raises an error
    y2 = np.random.random((np.random.randint(1, 10), n + np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        y2 @ x

    # test left multiplication by vector raises an error
    y3 = np.random.random(n)
    with pytest.raises(NotImplementedError):
        y3 @ x

    # test valid left multiplication by LowRank
    y4_nrows = np.random.randint(1, 10)
    y4_ncol = n
    y4_factor_ncol = np.random.randint(1, 10)
    y4_a = np.random.random((y4_nrows, y4_factor_ncol))
    y4_b = np.random.random((y4_ncol, y4_factor_ncol))
    y4 = LowRank(y4_a, y4_b)
    z4 = y4 @ x
    assert(np.allclose(z4.toarray(), y4.a.dot(y4.b.T.dot(x.toarray()))))

    # test invalid left multiplication by LowRank raises an error
    y5_nrows = np.random.randint(1, 10)
    y5_ncol = n + np.random.randint(1, 10)
    y5_factor_ncol = np.random.randint(1, 10)
    y5_a = np.random.random((y5_nrows, y5_factor_ncol))
    y5_b = np.random.random((y5_ncol, y5_factor_ncol))
    y5 = LowRank(y5_a, y5_b)
    with pytest.raises(ValueError):
        y5 @ x

def test_right_mul():
    # test valid right multiplication by matrix
    y1 = np.random.random((m, np.random.randint(1, 10)))
    z1 = x @ y1
    assert(np.allclose(z1.toarray(), x.a.dot(x.b.T).dot(y1)))

    # test invalid right multiplication by matrix raises an error
    y2 = np.random.random((m + np.random.randint(1, 10), np.random.randint(1, 10)))
    with pytest.raises(ValueError):
        x @ y2

    # test right multiplication by vector raises an error
    y3 = np.random.random(m)
    with pytest.raises(NotImplementedError):
        x @ y3

    # test valid right multiplication by LowRank
    y4_nrows = m
    y4_ncol = np.random.randint(1, 10)
    y4_factor_ncol = np.random.randint(1, 10)
    y4_a = np.random.random((y4_nrows, y4_factor_ncol))
    y4_b = np.random.random((y4_ncol, y4_factor_ncol))
    y4 = LowRank(y4_a, y4_b)
    z4 = x @ y4
    assert(np.allclose(z4.toarray(), x.toarray().dot(y4.a.dot(y4.b.T))))

    # test invalid right multiplication by LowRank raises an error
    y5_nrows = m + np.random.randint(1, 10)
    y5_ncol = np.random.randint(1, 10)
    y5_factor_ncol = np.random.randint(1, 10)
    y5_a = np.random.random((y5_nrows, y5_factor_ncol))
    y5_b = np.random.random((y5_ncol, y5_factor_ncol))
    y5 = LowRank(y5_a, y5_b)
    with pytest.raises(ValueError):
        x @ y5
