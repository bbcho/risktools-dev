import pandas as _pd
import numpy as _np
import statsmodels.formula.api as _smf
import ctypes
from numpy.ctypeslib import ndpointer
import os
import multiprocessing as mp
import time
from numpy.random import default_rng

def _import_csimOU():
    dir = os.path.dirname(os.path.realpath(__file__)) + "/c/"
    lib = ctypes.cdll.LoadLibrary(dir + "simOU.so")
    fun = lib.csimOU
    fun.restype = None
    fun.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    fun.restype = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
    return fun


def _simOUc(s0=5, theta=2, mu=4, dt=1/252, sigma=1, T=1, sims=10, eps=None):
    fun = _import_csimOU()

    # calc periods
    N = int(T/dt)
    
    # check on random number size
    if (N+1)*sims > 200_000_000:
        import warnings
        warnings.warn(
            '''
            Note that this simulation will generate more than
            200M random numbers which may crash the python kernel. It may
            be a better idea to split the simulation into a series of smaller
            sims and then join them after.
            '''
            )
    
    # make mu an array of same size as N+1
    try:
        iter(mu)
        if len(mu) != N+1:
            raise ValueError("if mu is passed as an iterable, it must be of length int(T/dt) + 1 to account for starting value s0")
    except:
        mu = _np.ones((N+1))*mu
    
    # make same size as 1D array of all periods and sims
    mu = _np.tile(_np.array(mu), sims)

    # generate a 1D array of random numbers that is based on a 
    # 2D array of size P x S where P is the number of time steps
    # including s0 (so N + 1) and S is the number of sims. 1D 
    # array will be of size P * S. This is actually the slowest
    # part.
    if eps is None:
        rng = default_rng()
        x = rng.normal(loc=0, scale=_np.sqrt(dt), size=((N+1)*sims))
        x[0] = s0
        x[::(N+1)] = s0
    else:
        x = eps.T
        x = x.reshape((N+1)*sims)
        x[:,0] = s0
    
    # run simulation directly in-place on memory to save time.
    fun(x, theta, mu, dt, sigma, N+1, sims)
    
    return _pd.DataFrame(x.reshape((sims, N+1)).T)



if __name__ == "__main__":
    import time
    import risktools as rt

    ss = 20000

    ST = time.time()
    df = rt.simOU(dt=1/252, T=15, sims=ss, c=True)
    print(time.time() - ST)

    ST = time.time()

    df = _simOUc(dt=1/252, T=15, sims=ss)

    print(time.time() - ST)

    print(df.iloc[:,:10])

