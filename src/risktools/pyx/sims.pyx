# https://blog.paperspace.com/boosting-python-scripts-cython/
# https://medium.com/towards-data-science/numpy-array-processing-with-cython-1250x-faster-a80f8b3caa52

import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
def csimOU(
    double[::1] x,
    double theta,
    double[::1] mu,
    double dt,
    double[::1] sigma,
    unsigned long long int rows,
    unsigned long long int cols,
    unsigned int log_price
    ):
    cdef long long int i = 1
    cdef long long int j = 0

    # pre-compute to make faster
    cdef long long int ll = rows * cols

    cdef double sq = np.sqrt(dt)

    # input x is a 2D array that has been reshaped to be 1D.
    # Loop through entire 1D array of length rows*cols,
    # skips value every time counter j is reaches row
    # count - effectively a new sim.

    if log_price != 0:
        for i in range(1, ll):
            if j >= (cols - 1):
                j = 0
            else:
                j = j + 1
                x[i] = x[i - 1] + (theta * (mu[i] - x[i - 1]) - 0.5 * sigma[i] * sigma[i]) * dt + sigma[i] * sq * x[i];
    else:
        for i in range(1, ll):
            if j >= (cols - 1):
                j = 0
            else:
                j = j + 1
                x[i] = x[i - 1] + (theta * (mu[i] - x[i - 1])) * dt + sigma[i] * sq * x[i];  

    return np.asarray(x)



@cython.boundscheck(False)
@cython.wraparound(False)
def csimOUJ(
    double[::1] x,
    double[::1] elp,
    double[::1] ejp,
    double theta,
    double[::1] mu,
    double dt,
    double[::1] sigma,
    unsigned long long int rows,
    unsigned long long int cols,
    unsigned int mr_lag,
    double jump_prob,
    double jump_avgsize
    ):
    cdef long long int i = 1
    cdef long long int j = 0
    cdef long long int end

    # pre-compute to make faster
    cdef long long int ll = rows * cols

    cdef double sq = np.sqrt(dt)

    # input x is a 2D array that has been reshaped to be 1D.
    # Loop through entire 1D array of length rows*cols,
    # skips value every time counter j is reaches row
    # count - effectively a new sim.

    for i in range(1, ll):
        if j >= (cols - 1):
            j = 0
        else:
            j = j + 1

            # calc step
            x[i] = (
                x[i - 1]
                + theta
                    * (mu[i] - jump_prob * jump_avgsize - x[i - 1])
                    * x[i - 1]
                    * dt
                + sigma[i] * x[i - 1] * x[i] * sq
                + ejp[i] * elp[i]
            )

            if (ejp[i] > 0.0):
                # if there is a jump in this step, add it to the mean reversion
                # level so that it doesn't drop back down to the given mean too
                # quickly. Simulates impact of lagged market response to a jump

                # make sure that it doesn't roll over into a new simulation
                end = min(mr_lag, cols - j - 1)

                for k in range(i, i + end):
                    mu[k] = mu[k] + ejp[i] * elp[i]
                    if k > i:
                        ejp[k] = 0.0 # stops double jumps

    return np.asarray(x)
