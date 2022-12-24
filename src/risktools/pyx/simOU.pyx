# https://blog.paperspace.com/boosting-python-scripts-cython/
# https://medium.com/towards-data-science/numpy-array-processing-with-cython-1250x-faster-a80f8b3caa52

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def csimOU(
    double[::1] x,
    double theta,
    double[::1] mu,
    double dt,
    double sigma,
    unsigned long long int rows,
    unsigned long long int cols,
    unsigned int log_price
    ):
    cdef long long int i = 1
    cdef long long int j = 0

    # pre-compute to make faster
    cdef long long int ll = rows * cols
    cdef double ss = 0

    if log_price != 0:
        ss = 0.5 * sigma * sigma
    
    cdef double sq = np.sqrt(dt)

    # input x is a 2D array that has been reshaped to be 1D.
    # Loop through entire 1D array of length rows*cols,
    # skips value every time counter j is reaches row
    # count - effectively a new sim.

    for i in range(1, ll):
        if j >= (cols - 1):
            j = 0
        else:
            j = j + 1;
            x[i] = x[i - 1] + (theta * (mu[i] - x[i - 1]) - ss) * dt + sigma * sq * x[i];

    return np.asarray(x)
