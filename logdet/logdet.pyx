# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

"""
A run-time optimized, lapack-dependent module to compute the
log-determinant of a square symetric matrix (e.g. a variance-covariance
matrix).

Copyright (C) 2015 Oskar Maier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

##########
# Changelog
# 2015-06-01 released
# 2015-05-17 created
##########

# docstring info
__author__ = "Oskar Maier"
__copyright__ = "Copyright 2015, Oskar Maier"
__version__ = "0.1.0"
__maintainer__ = "Oskar Maier"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"

# python imports
import numpy as np

# cython imports
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log, sqrt

# code
def logdet(DTYPE_t[::1] AX not None):
    r"""
    Compute the log-determinant of the square symetric matrix X.
    
    This lapack-based version is roughly 4-times faster than using numpy.
    
    Parameters
    ----------
    AX : ndarray
        Upper or lower triangle of the square symetric matrix X, packed
        rowwise C-ordered) in a linear array.
    
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.random.rand(100, 5)
    >>> X = np.cov(data, rowvar=0, ddof=1)
    >>> AX = X[np.triu_indices(X.shape[0])]
    >>> import logdet
    >>> logdet.logdet(AX)
    -12.449708530090223
    
    >>> np.log(np.linalg.det(X))
    -12.449708530090222
    """
    cdef SIZE_t l = AX.shape[0]
    cdef SIZE_t n = <SIZE_t>round(-.5 + sqrt(.25 + 2 * l))
    return logdetUDU(&AX[0], n)

cdef DTYPE_t logdet_(DTYPE_t* AX, SIZE_t n) nogil:
    r"""
    Compute the log-determinant of the square symetric matrix X.
    This is the gil-free cython version of the logdet python function.
    
    Parameters
    ----------
    AX : ndarray
        Upper or lower triangle of the square symetric matrix X, packed
        rowwise (C-ordered) in a linear array.
    n : int
        Side-length of square symetric matrix X.
    """
    return logdetUDU(AX, n)

cdef DTYPE_t logdetUDU(DTYPE_t* X, SIZE_t n) nogil:
    r"""
    Computes the log-determinant of a symmetric positive
    semi-definite matrix stored as condensed form upper triangular matrix in
    C order (row-major) using UDU decomposition.
    
    Numerical instability of the UDU decomposition or a singular input matrix X
    can cause the determinant to be <=0. This case is intercepted at a threshold
    [SINGULARITY_THRESHOLD] and log(SINGULARITY_THRESHOLD) returned instead.
    Therefore the returned value is always >= log(SINGULARITY_THRESHOLD).
    
    Note: X is expected to be a packed upper triangular matrix in C-order (row-major)
          X will be copied and transfered to Fortran order
          n denotes the nxn dimension of the original symmetric positive semi-definite matrix behind X
    """
    # Thanks to: http://biostatmatt.com/archives/17
    cdef:
        int i, info
        int* ipiv
        double logdet, tmp
        double* Xf
    
    # allocate memory
    ipiv = <int*>malloc(n * sizeof(int))
    if NULL == ipiv:
        with gil: raise MemoryError()
    
    # to fortran order (column-major), copies data
    Xf = asfortran(X, n)
    
    # solves A = U*D*U**T => output A = D
    dsptrf_('U', <int*>&n, Xf, ipiv, &info)

    # Interpretation of info:
    # info == 0: all is fine
    # info > 0:  system is singular (D has a 0 on the diagonal), mutliplication with 0 will occur for det, det will be 0
    # info < 0:  invalid argument at position abs(info)
    
    # assemble logdet
    logdet = 0.
    for i in range(n):
        if ipiv[i] > 0:
            tmp = Xf[ umat(i,i) ]
            if tmp < SINGULARITY_THRESHOLD: return log(SINGULARITY_THRESHOLD)
            logdet += log(tmp)
        elif i > 0 and ipiv[i] < 0 and ipiv[i-1] == ipiv[i]:
            tmp = Xf[ umat(i,i) ] * Xf[ umat(i-1,i-1) ] -\
                   Xf[ umat(i-1,i) ] * Xf[ umat(i-1,i) ]
            if tmp < SINGULARITY_THRESHOLD: return log(SINGULARITY_THRESHOLD)
            logdet += log(tmp)
        
    free(ipiv)
    free(Xf)
    
    return <DTYPE_t>logdet if logdet > log(SINGULARITY_THRESHOLD) else <DTYPE_t>log(SINGULARITY_THRESHOLD)

cdef inline DTYPE_t* asfortran(DTYPE_t* C, SIZE_t n) nogil:
    """
    Takes a condensed form upper triangular matrix in
    C order (row-major) and converts it to fortran order
    (column-major) in condensed format.
    """
    cdef:
        SIZE_t m = upper_n_elements(n)
        DTYPE_t* F
        SIZE_t i, j
        
    F = <DTYPE_t*>malloc(m * sizeof(double))
    if NULL == F:
        with gil: raise MemoryError()
    
    for j in range(n):
        for i in range(n - j):
            F[idx_f(i, j)] = C[idx_c(i, j, n)]
    
    return F

cdef inline SIZE_t idx_f(SIZE_t i, SIZE_t j) nogil:
    """
    Compute the fortran-style index of a condensed upper triangular matrix.
    Assuming (slower) j<-0:n and (faster) i<-0:n-j.
    """
    return j + (i + j) * (i + j + 1) / 2

cdef inline SIZE_t idx_c(SIZE_t i, SIZE_t j, SIZE_t n) nogil:
    """
    Compute the C-style index of a condensed upper triangular matrix.
    Assuming (slower) j<-0:n and (faster) i<-0:n-j.
    """
    return j * n - j * (j + 1) / 2 + i + j

cdef inline int umat(int i, int j) nogil:
    return i + j * ( j + 1 ) / 2

cdef inline SIZE_t upper_n_elements(SIZE_t n) nogil:
    "The number of (diagonal including) elements of an upper triangular nxn matrix."
    return (n * n + n) / 2
