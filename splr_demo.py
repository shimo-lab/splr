# import splr
# import scipy.sparse
# import scipy.linalg
# import numpy as np
#
# m = 1000
# n = 1000
#
# x = scipy.sparse.rand(m, n, density=0.0003) # creation of a random sparse matrix
# y = splr.LowRank(np.ones((m, 1)), np.ones((n, 1))) # creation of a rank 1 matrix
# z = splr.SparsePlusLowRank(x, y) # creation of the matrix of interest
#
# k = 200 # number of singular values / vectors to compute
# threshold = 0.01
# max_iter = 10000
#
# u, d, v = z.svd(k, threshold, max_iter) # computation of the truncated SVD
#
# z_array = z.sparse.toarray() + z.low_rank.toarray() # explicit computation of z
# z_approx = (u * d).dot(v.T) # approximation of z by its truncated SVD
#
# print("Rank of the original matrix: {} \n".format(np.linalg.matrix_rank(z_array)))
#
# print("Relative difference between the original matrix and its truncated SVD \n" +
#       "approximation (L2 norm): {}".format(np.linalg.norm(z_array - z_approx) / np.linalg.norm(z_array)))

from splr import LowRank

import numpy as np


n = np.random.randint(1, 10)
m = np.random.randint(1, 10)
p = np.random.randint(1, 10)

a = np.random.random((n, p))
b = np.random.random((m, p))
x = LowRank(a, b)

y1 = np.random.random((np.random.randint(1, 10), n))
z1 = (y1 @ x).toarray()
