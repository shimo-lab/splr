# Sparse plus low rank module
<br/>
Implementation of two classes to handle matrices that can be written as the sum of a sparse matrix and a low rank
matrix.
<br/>
<br/>
The main motivation for this is to be able to efficiently handle sparse data matrices
after centering.<br/><br/>
 
 ## Supported operations
 * Low rank matrices : right and left multiplication by another matrix
 <br/><br/>
 * Sparse plus low rank matrices : right and left multiplication by another matrix,
  and rank-restricted singular value decomposition (SVD)<br/><br/>
 
## `pip` installation 
Run the following command :

```
pip install -e git+https://github.com/shimo-lab/splr#egg=splr
```
<br/>

## References
* Hastie et al., matrix completion and low-rank SVD via fast alternating least square,
Journal of Machine Learning Research, 2015 <br/><br/>
* Hastie et al., softImpute : matrix completion via iterative soft-thresholded SVD,
R package, 2015