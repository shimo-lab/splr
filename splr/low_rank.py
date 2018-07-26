"""
Definition of a class to handle low rank matrices in a factorized form
"""

import numpy as np


def is_low_rank(x):
    """
    Function testing the input is an instance of LowRank

    Parameters
    ----------
    x
        Object to check for being a low rank matrix

    Returns
    -------
    bool
        True if x is a low rank matrix, False otherwise

    """
    return isinstance(x, LowRank)


class LowRank:
    """
    Low rank matrix class, which allows to represent the low rank matrix a @ b.T efficiently

    Parameters
    ----------
    a : ndarray
    b : ndarray

    """

    __array_priority__ = 1000 # sets priority in order for numpy to call custom operators

    def __init__(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.ndim == 2 and b.ndim == 2:
                if a.shape[1] == b.shape[1]:
                    self.a = a
                    self.b = b
                    self.shape = (a.shape[0], b.shape[0])
                else:
                    raise ValueError("inconsistent shapes")

            else:
                raise NotImplementedError("current implementation only supports 2D arrays")

        else:
            raise ValueError("arguments must be numpy.ndarray")

    def __neg__(self):
        """Opposite operator"""
        return LowRank(- self.a, self.b)

    def __getattr__(self, attr):
        """Implements self.T alias for self.transpose()"""
        if attr == 'T':
            return self.transpose()
        else:
            raise AttributeError(attr + " not found")

    def __matmul__(self, other):
        """
        Matrix right multiplication operator

        Parameters
        ----------
        other : LowRank or numpy.ndarray
            The low rank matrix with which to perform the multiplication

        Returns
        -------
        LowRank :
            The multiplication's result

        """
        if is_low_rank(other):
            if self.shape[1] == other.shape[0]:
                # TODO: think about efficiency
                return LowRank(self.a.dot(self.b.T.dot(other.a)), other.b)
            else:
                raise ValueError("inconsistent shapes")

        elif isinstance(other, np.ndarray):
            if other.ndim == 2:
                if self.shape[1] == other.shape[0]:
                    return LowRank(self.a, other.T.dot(self.b))
                else:
                    raise ValueError("inconsistent shapes")

            else:
                raise NotImplementedError("current implementation only support 2D arrays")

        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """Matrix left multiplication operator"""

        if is_low_rank(other):
            if other.shape[1] == self.shape[0]:
                # TODO: think about efficiency
                return LowRank(other.a.dot(other.b.T.dot(self.a)), self.b)
            else:
                raise ValueError("inconsistent shapes")

        elif isinstance(other, np.ndarray):
            if other.ndim == 2:
                if other.shape[1] == self.shape[0]:
                    return LowRank(other.dot(self.a), self.b)
                else:
                    raise ValueError("inconsistent shapes")

            else:
                raise NotImplementedError("current implementation only support 2D arrays")

        else:
            return NotImplemented

    def transpose(self):
        """Transposition method"""
        return LowRank(self.b, self.a)

    def toarray(self):
        """
        Method performing the conversion to ndarray

        Returns
        -------
        ndarray
            Result of the right multiplication of the first factor by the transpose of the second factor

        """
        return self.a.dot(self.b.T)