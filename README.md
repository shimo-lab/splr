# Sparse plus low rank module (under development)
<br/>
Implementation of two classes to handle matrices that can be written as the sum of a sparse matrix and a low rank
matrix.
<br/>
<br/>
Please note that this library is still under active development. It has only been tested on a small number of cases, and further experiments must be conducted to ensure its proper functioning. 
<br/><br/>
 
 ## Supported operations
 * Low rank matrices : right and left multiplication by another matrix
 <br/><br/>
 * Sparse plus low rank matrices : right and left multiplication by another matrix, 
 rank-restricted singular value decomposition (SVD)<br/><br/>
 
## `pip` installation 
Run the following command :

```
pip install -e git+https://github.com/shimo-lab/splr#egg=splr
```
<br/>

## Tests
Run ```pytest``` to run the tests.<br/><br/>

## References
* Hastie et al., matrix completion and low-rank SVD via fast alternating least square,
Journal of Machine Learning Research, 2015 <br/><br/>
* Hastie et al., softImpute : matrix completion via iterative soft-thresholded SVD,
R package, 2015
