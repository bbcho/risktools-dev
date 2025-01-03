# https://blog.paperspace.com/boosting-python-scripts-cython/
# https://medium.com/towards-data-science/numpy-array-processing-with-cython-1250x-faster-a80f8b3caa52

import numpy as np
cimport numpy as np
cimport cython

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
            j = j + 1;
            x[i] = x[i - 1] + (theta * (mu[i] - x[i - 1])) * dt + sigma[i] * sq * x[i];


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

            if (mr_lag is not None) & (ejp[i] > 0):
                # if there is a jump in this step, add it to the mean reversion
                # level so that it doesn't drop back down to the given mean too
                # quickly. Simulates impact of lagged market response to a jump

                # make sure that it doesn't roll over into a new simulation
                end = min(mr_lag, cols - j - 1)

                for k in range(i, i + end):
                    mu[k] = mu[k] + ejp[k] * elp[k]


    return np.asarray(x)
