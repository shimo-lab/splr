"""
LowRank unit tests
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
    z1 = x.left_mul(y1)
    assert(np.allclose(z1, y1.dot(x.a).dot(x.b.T)))

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
    assert(np.allclose(z1, x.a.dot(x.b.T).dot(y1)))

    # test invalid right multiplication by matrix raises an error
    y2 = np.random.random((m + np.random.randint(1, 10), np.random.randint(1, 10)))
    with pytest.raises(AssertionError):
        x.right_mul(y2)

    # test right multiplication by vector raises an error
    y3 = np.random.random(m)
    with pytest.raises(AssertionError):
        x.right_mul(y3)
