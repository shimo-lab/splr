import time
import splr
import scipy.sparse
import scipy.linalg
import numpy as np


n = 3000 # matrix size
m = n
density = 1e-4 # density of non-zero coefficients

# creation and centering of the matrix for which to perform SVD
x = scipy.sparse.rand(n, m, density=density)
x_centered = splr.center_matrix(x)

k = int(scipy.sqrt(n * m * density)) # number of singular values to compute
alpha = 0.0 # regularization parameter

# splr SVD
start_time = time.time()
u, d, v = x_centered.svd(k, alpha)
end_time = time.time()

splr_svd_duration = end_time - start_time

# scipy SVD for dense matrices
x_centered_dense = x_centered.toarray()

start_time = time.time()
u_dense, d_dense, tv_dense = scipy.linalg.svd(x_centered_dense)
end_time = time.time()

scipy_svd_duration = end_time - start_time

# approximation error for scipy SVD
err_dense = np.linalg.norm(x_centered_dense - (u_dense[:, :k] * d_dense[:k]) @ tv_dense[:k, :])

# approximation error for splr SVD
err_splr = np.linalg.norm(x_centered_dense - (u * d) @ v.T)

# excess approximation error for splr SVD
ex_rel_approx_err = max(0, (err_splr - err_dense) / err_dense)

print("splr SVD computed in {:.2f} seconds".format(splr_svd_duration))
print("scipy SVD computed in {:.2f} seconds".format(scipy_svd_duration))
print("excess relative approximation error : {0:.2%}".format(ex_rel_approx_err))
