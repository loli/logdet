# python imports
import numpy as np

# cython imports
cimport numpy as np

# type definitions
ctypedef np.npy_float64 DTYPE_t # data type
ctypedef np.npy_intp SIZE_t # type for indices and counters

# constants
cdef double SINGULARITY_THRESHOLD = 1e-6

# extern cdefs: lapack c-wrapped fortran routine definitions
# A = U*D*U**T // http://www.netlib.org/lapack/explore-html/d1/dcd/dsptrf_8f.html
cdef extern void dsptrf_( char *uplo, int *n, double *ap, int *ipiv, int *info ) nogil
# A = P*L*U // http://www.netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f.html
cdef extern void dgetrf_( int *m, int *n, double *a, int *lda, int *ipiv, int *info ) nogil

# public methods
cdef DTYPE_t logdet_(DTYPE_t* AX, SIZE_t n) nogil