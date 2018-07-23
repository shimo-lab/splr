"""
Definition of two classes to handle matrices written as the sum of a sparse and a low rank matrix
"""

import scipy.sparse


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


class LowRank:
    """
    Low rank matrix class

    Parameters
    ----------
    a : ndarray, shape (m, p)
    b : ndarray, shape (n, p)
    """

    def __init__(self, a, b):
        assert(a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1]) # check input arguments have proper shape
        self.a = a
        self.b = b
        self.shape = (a.shape[0], b.shape[0])

    def __eq__(self, other):
        """
        Function testing the equality of two LowRank instances

        Parameters
        ----------
        other
            Object to check for being equal to self

        Returns
        -------
        bool
            True if other is equal to self, False otherwise

        """
        if is_low_rank(other):
            return (self.a == other.a and self.b == other.b)
        else:
            return False

    def right_mul(self, mat):
        """
        Function performing right multiplication by a matrix

        Parameters
        ----------
        mat : ndarray
            The matrix by which to perform right multiplication

        Returns
        -------
        ndarray
            The multiplication's result

        """
        assert(mat.ndim == 2 and mat.shape[0] == self.shape[1]) # check input argument have proper shape
        return self.a.dot(self.b.T.dot(mat))

    def left_mul(self, mat):
        """
        Function performing left multiplication by a matrix

        Parameters
        ----------
        mat : ndarray
            The matrix by which to perform left multiplication

        Returns
        -------
        ndarray
            The multiplication's result

        """
        assert(mat.ndim == 2 and mat.shape[1] == self.shape[0]) # check input argument have proper shape
        return (mat.dot(self.a)).dot(self.b.T)

    def eval(self):
        """
        Function performing the evaluation of the low rank matrix

        Returns
        -------
        ndarray
            Result of the multiplication of self.a by the transpose of self.b

        """
        return self.a.dot(self.b.T)


class SparsePlusLowRank:
    """
    Sparse plus low rank matrix class

    Parameters
    ----------
    sparse : scipy.sparse.spmatrix, shape (m, n)
    low_rank : LowRank, shape (m, n)
    """

    def __init__(self, sparse, low_rank):
        assert(scipy.sparse.issparse(sparse) and is_low_rank(low_rank)) # check input arguments belong to proper classes
        assert(sparse.shape == low_rank.shape) # check input arguments have compatible shapes
        self.sparse = sparse
        self.low_rank = low_rank
        self.shape = sparse.shape

    def __eq__(self, other):
        """
        Function testing the equality of two SparsePlusLowRank instances

        Parameters
        ----------
        other
            Object to check for being equal to self

        Returns
        -------
        bool
            True if other is equal to self, False otherwise

        """
        if is_sparse_plus_low_rank(other):
            return (self.sparse == other.sparse and self.low_rank == other.low_rank)
        else:
            return False

    def right_mul(self, mat):
        """
        Function performing right multiplication by a matrix

        Parameters
        ----------
        mat : ndarray
            The matrix by which to perform right multiplication

        Returns
        -------
        ndarray
            The multiplication's result

        """
        assert(mat.ndim == 2 and mat.shape[0] == self.shape[1]) # check input argument have proper shape
        res_sparse = self.sparse * mat
        res_low_rank = self.low_rank.right_mul(mat)
        return res_sparse + res_low_rank

    def left_mul(self, mat):
        """
        Function performing left multiplication by a matrix

        Parameters
        ----------
        mat : ndarray
            The matrix by which to perform left multiplication

        Returns
        -------
        ndarray
            The multiplication's result

        """
        assert(mat.ndim == 2 and mat.shape[1] == self.shape[0]) # check input argument have proper shape
        res_sparse = mat * self.sparse
        res_low_rank = self.low_rank.left_mul(mat)
        return res_sparse + res_low_rank

    def svd(self):
        """
        To be implemented ...
        """
        pass
