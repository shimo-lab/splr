"""
Definition of a class to handle matrices written as the sum of a sparse and a low rank matrix
"""

import scipy.linalg
import scipy.sparse
import numpy as np

from .low_rank import is_low_rank, LowRank


def is_sparse_plus_low_rank(x):
    """
    Function testing the input is an instance of SparsePlusLowRank

    Parameters
    ----------
    x
        Object to check for being the sum of sparse and low rank matrices

    Returns
    -------
    bool
        True if x is the sum of sparse and low rank matrices, False otherwise

    """
    return isinstance(x, SparsePlusLowRank)


class SparsePlusLowRank:
    """
    Sparse plus low rank matrix class

    Parameters
    ----------
    sparse : scipy.sparse.spmatrix
    low_rank : LowRank

    """

    __array_priority__ = 1000  # sets priority in order for numpy to call custom operators

    def __init__(self, sparse, low_rank):
        if scipy.sparse.issparse(sparse) and is_low_rank(low_rank):
            if sparse.shape == low_rank.shape:
                self.sparse = sparse
                self.low_rank = low_rank
                self.shape = sparse.shape
            else:
                raise ValueError("inconsistent shapes")

        else:
            raise TypeError("arguments must be of type scipy.sparse.spmatrix and LowRank")

    def __neg__(self):
        """Opposite operator"""
        return SparsePlusLowRank(- self.sparse, - self.low_rank)

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
        other : ndarray
            The matrix with which to perform the multiplication

        Returns
        -------
        ndarray :
            The multiplication's result

        """
        def check_shape(other):
            if self.shape[1] != other.shape[0]:
                raise ValueError("inconsistent shapes")

        if isinstance(other, np.ndarray):
            if other.ndim == 2:
                check_shape(other)
                return self.sparse @ other + (self.low_rank @ other).toarray()
            else:
                raise NotImplementedError("current implementation only support 2D arrays")

        else:
            raise TypeError("argument must be ndarray")

    def __rmatmul__(self, other):
        """Matrix left multiplication operator"""

        def check_shape(other):
            if other.shape[1] != self.shape[0]:
                raise ValueError("inconsistent shapes")

        if isinstance(other, np.ndarray):
            if other.ndim == 2:
                check_shape(other)
                return other @ self.sparse + (other @ self.low_rank).toarray()
            else:
                raise NotImplementedError("current implementation only support 2D arrays")

        else:
            raise TypeError("argument must be ndarray")

    def transpose(self):
        """Transposition method"""
        return SparsePlusLowRank(self.sparse.T, self.low_rank.T)

    def toarray(self):
        """
        Method performing the conversion to ndarray

        Returns
        -------
        ndarray
            Result of the addition of the sparse and low rank matrices

        """
        return self.sparse.toarray() + self.low_rank.toarray()

    def svd(self, k, threshold=0.01, max_iter=1000):
        """
        Method computing the truncated singular value decomposition by alternating regression
        TODO: implement regularization (ridge regression)

        Parameters
        ----------
        k : int
            Number of singular values / vectors to compute
        threshold : float
            Convergence threshold for the Frobenius norm ratio between two consecutive step's solution
        max_iter : int
            Maximum number of iterations

        Returns
        -------
        u : ndarray, shape=(m, k)
            Left singular vectors
        d : ndarray, shape=(k,)
            Singular values
        v : ndarray, shape=(n, k)
            Right singular vectors

        """
        # Initialization TODO: clarify the choice of these variables' initial value
        u = np.random.multivariate_normal(np.zeros(k), np.eye(k), size=self.shape[0])
        v = np.zeros((self.shape[1], k))
        d_square = np.ones(k)

        ratio = 1 # used to check convergence
        curr_iter = 0

        while ratio > threshold and curr_iter < max_iter:
            curr_iter += 1

            # saving the old state to check convergence
            old_u = u
            old_v = v
            old_d_square = d_square

            # v update step
            b = self.T @ u
            v, d_square, _ = scipy.linalg.svd(b, full_matrices=False)

            # u update step
            a = self @ v
            u, d_square, _ = scipy.linalg.svd(a, full_matrices=False)

            # convergence check
            old_frob_norm = (old_d_square ** 2).sum()
            frob_norm = (d_square ** 2).sum()
            frob_inner_prod = np.diag((old_d_square * old_u.T.dot(u)).dot(d_square * v.T.dot(old_v))).sum()
            ratio = (frob_norm + old_frob_norm - 2 * frob_inner_prod) / old_frob_norm

        if curr_iter == max_iter:
            print("Warning: convergence not achieved")

        # final cleanup
        m = self @ v
        u, d, rt = scipy.linalg.svd(m, full_matrices=False)

        return u, np.maximum(d, 0), v.dot(rt.T)