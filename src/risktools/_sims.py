import pandas as _pd
import numpy as _np
import statsmodels.formula.api as _smf
import ctypes
from numpy.ctypeslib import ndpointer
import os
import multiprocessing as mp
import time
from numpy.random import default_rng, Generator, SFC64
import platform
from .extensions import csimOU as _csimOU
from .extensions import csimOUJ as _csimOUJ


class Result:
    def __init__(self):
        self.val = _pd.DataFrame()

    def update_result(self, val):
        self.val = _pd.concat([self.val, val], axis=1)  # append by columns


def is_iterable(x):
    try:
        iter(x)
        return True
    except:
        return False


def make_into_array(x, N):
    # make an array of same size as N+1
    if is_iterable(x):
        if len(x.shape) == 2:
            # if a 2D array is passed, return it as is
            # good for stocastic volatility matrix
            x = _np.vstack((x[0], x))
            return x

        if x.shape[0] == N:
            x = _np.append(x[0], x)
        else:
            raise ValueError(
                "if mu is passed as an iterable, it must be of length int(T/dt)"
            )

    else:
        x = _np.ones(N + 1) * x

    return x


def simGBM(s0=10, mu=0, sigma=0.2, r=0, T=1, dt=1 / 252, sims=1000, eps=None):
    """
    Simulates a Geometric Brownian Motion stochastic process (random walk)

    Parameters
    ----------
    s0 : float
        Starting value for GBM process at time = 0. By default, this is 10.
    mu : float
        Mean for normally distribution for normally distributed returns.
        Not used if eps is provided. By default, this is 0.
    sigma : float
        Annualized standard deviation for normally distributed returns. By default, this is 0.2.
        Not used if eps is provided.
    r : float
        Interest rate to discount the process by. By default, this is 0.
    T : float
        Time to maturity in years. But default, this is 1.
    dt : float
        Time step in period e.g. 1/252 = 1 business day. By default, this is 1/252.
    sims : int
        Number of simulations to run. By default, this is 1000.
    eps : numpy array
        Random numbers to use for the returns. If provided, mu and sigma are ignored.
        Must of size (p x sims) where p is the number of periods in T/dt.

    Returns
    -------
    A list of simulated values

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simGBM(s0=5, mu=0, sigma=0.2, r=0.01, T=2, dt=1/252, sims=1000)
    """

    periods = int(T / dt)
    s = _np.zeros((periods + 1, sims))
    s = _pd.DataFrame(data=s)

    if eps is None:
        eps = _np.random.normal(mu, 1, size=(periods, sims))

    s.iloc[0, :] = s0
    s.iloc[1:, :] = eps

    # calc geometric brownian motion
    s.loc[1:, :] = _np.exp(
        (r - sigma**2 / 2) * dt + sigma * _np.sqrt(dt) * s.loc[1:, :]
    )

    s = s.cumprod()

    return s


def _import_csimOU():
    dir = os.path.dirname(os.path.realpath(__file__)) + "/../"  # + "/c/"

    ext = ".so"
    if platform.system() == "Windows":
        ext = ".dll"

    lib = ctypes.cdll.LoadLibrary(dir + "simOU" + ext)
    fun = lib.csimOU
    fun.restype = None
    fun.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    fun.restype = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
    return fun


def simOU(
    s0=5,
    mu=4,
    theta=2,
    sigma=1,
    T=1,
    dt=1 / 252,
    sims=1000,
    eps=None,
    seed=None,
    log_price=False,
    c=True,
):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk) with multiple
    simulations

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float | array-like (1D or 2D)
        Mean that the stochastic diffusion equation will revert to. Scalars used for constants means through
        time and through various simulations. 1D arrays are supported for time varying means which must be
        the same length as T/dt (i.e. the number of periods). 2D arrays are also supported for stochastic
        mu where the first dimension is the number of periods and the second dimension equals the number of
        simulations.
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float | array-like (1D or 2D)
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt).
        1D arrays are supported for time varying volatility which must be the same length as T/dt (i.e. the number of
        periods). 2D arrays are also supported for stochastic volatility where the first dimension is the number of
        periods and the second dimension equals the number of simulations.
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run. By default, this is 1000. Not used if eps is provided.
    eps : matrix-like[float]
        Random numbers to use for the returns. If provided, mu, sigma, T, dt and sims are ignored.
        Must of size (p x sims) where p is the number of periods in T, i.e. int(T/dt).
        Excludes time 0.
    seed : int
        To pass to numpy random number generator as seed. For testing only.
    log_price : bool
        Adds adjustment term to the mean reversion term if the prices passed are log prices. By
        default False.
    c : bool
        Whether or not to run C optimized code. By default True. Otherwise use python loop.

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU()
    """
    if eps is not None:
        sims = eps.shape[1]

    # number of business days in a year
    bdays_in_year = 252

    # calc periods
    N = int(T / dt)

    # print half-life of theta
    print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)

    # make mu array
    mu = make_into_array(mu, N)
    sigma = make_into_array(sigma, N)

    # make same size as 1D array of all periods and sims
    if len(mu.shape) == 1:
        mu = _np.tile(_np.array(mu), sims)
    else:
        mu = mu.flatten("F")

    # Don't run if 2D array passed for sigma
    if len(sigma.shape) == 1:
        sigma = _np.tile(_np.array(sigma), sims)
    else:
        sigma = sigma.flatten("F")

    if c == True:
        return _simOUc(
            s0=s0,
            mu=mu,
            theta=theta,
            T=T,
            dt=dt,
            sigma=sigma,
            sims=sims,
            eps=eps,
            seed=seed,
            log_price=log_price,
        )
    else:
        return _simOUpy(
            s0=s0,
            mu=mu,
            theta=theta,
            T=T,
            dt=dt,
            sigma=sigma,
            sims=sims,
            eps=eps,
            seed=seed,
            log_price=log_price,
        )


def _simOUc(s0, theta, mu, dt, sigma, T, sims=10, eps=None, seed=None, log_price=False):
    # calc periods
    N = int(T / dt)

    # check on random number size
    if (N + 1) * sims > 200_000_000:
        import warnings

        warnings.warn(
            """
            Note that this simulation will generate more than
            200M random numbers which may crash the python kernel. It may
            be a better idea to split the simulation into a series of smaller
            sims and then join them after.
            """
        )

    # generate a 1D array of random numbers that is based on a
    # 2D array of size P x S where P is the number of time steps
    # including s0 (so N + 1) and S is the number of sims. 1D
    # array will be of size P * S. This is actually the slowest
    # part.
    if eps is None:
        # rng = default_rng(seed)
        rng = Generator(SFC64(seed))
        x = rng.normal(loc=0, scale=1, size=((N + 1) * sims))
        x[0] = s0
        x[:: (N + 1)] = s0
    else:
        x = eps.T
        x = _np.c_[_np.ones(sims) * s0, x]
        x = x.reshape((N + 1) * sims)

    x = _csimOU(
        x, theta, mu, dt, sigma, rows=sims, cols=N + 1, log_price=int(log_price)
    )

    return _pd.DataFrame(x.reshape((sims, N + 1)).T)


def _simOUpy(
    s0, mu, theta, sigma, T, dt, sims=1000, eps=None, seed=None, log_price=False
):

    # number of periods dt in T
    N = int(T / dt)

    mu = mu.reshape((sims, N + 1)).T
    sigma = sigma.reshape((sims, N + 1)).T

    mu = _pd.DataFrame(mu)
    sigma = _pd.DataFrame(sigma)

    # init df with zeros, rows are steps forward in time, columns are simulations
    out = _np.zeros((N + 1, sims))
    out = _pd.DataFrame(data=out)

    # set first row as starting value of sim
    out.loc[0, :] = s0

    # calc gaussian vector
    if eps is None:
        # rng = default_rng(seed)
        rng = Generator(SFC64(seed))
        eps = rng.normal(size=(N, sims))

    out.iloc[1:, :] = eps

    if log_price:
        ss = 0.5 * sigma * sigma
        for i in range(1, N + 1):
            # calc step
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + (theta * (mu.iloc[i, :] - out.iloc[i - 1, :]) - ss.iloc[i, :]) * dt
                + sigma.iloc[i, :] * out.iloc[i, :] * _np.sqrt(dt)
            )
    else:
        for i in range(1, N + 1):
            # calc step
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + (theta * (mu.iloc[i, :] - out.iloc[i - 1, :])) * dt
                + sigma.iloc[i, :] * out.iloc[i, :] * _np.sqrt(dt)
            )

    return out


def simOUJ(
    s0=5,
    mu=5,
    theta=0.5,
    sigma=0.2,
    jump_prob=0.05,
    jump_avgsize=3,
    jump_stdv=0.05,
    T=1,
    dt=1 / 12,
    sims=1000,
    mr_lag=None,
    eps=None,
    elp=None,
    ejp=None,
    seed=None,
    c=True,
):
    """
    Function for calculating an Ornstein-Uhlenbeck Jump Mean Reversion stochastic process (random walk) with multiple
    simulations

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float, int or pandas Series
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods)
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float | array-like (1D or 2D)
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt).
        1D arrays are support for time varying volatility which must be the same length as T/dt (i.e. the number of
        periods). 2D arrays are also supported for stochastic volatility where the first dimension is the number of
        periods and the second dimension is the number of simulations.
    jump_prob : float
        Probablity of jumps for a Possion process.
    jump_avgsize : float
        Average size of jumps for a log normal distribution
    jump_stdv : float
        Standard deviation of average jump size for a log normal distribution
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run
    mr_lag : int, optional
        Lag in mean reversion. If None, then no lag is used. If > 0, then the diffusion does not immediately
        return the mean after a jump at theta but instead with remain near the jump level for mr_lag periods.
        By default, this is None.
    eps : numpy array, optional
        Array of random numbers to use for the simulation. If None, then random numbers are generated.
        By default, this is None.
    elp : numpy array, optional
        Array of random numbers to use for the log price jump. If None, then random numbers are generated.
        By default, this is None.
    ejp : numpy array, optional
        Array of random numbers to use for the jump size. If None, then random numbers are generated.
        By default, this is None.
    seed : int, optional
        To pass to numpy random number generator as seed. For testing only.
    log_price : bool, optional
        Adds adjustment term to the mean reversion term if the prices passed are log prices. By
        default False.
    c : bool, optional
        Whether or not to run C optimized code. By default True. Otherwise use python loop.

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOUJ()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    N = int(T / dt)

    # check on random number size
    if (N + 1) * sims > 200_000_000:
        import warnings

        warnings.warn(
            """
            Note that this simulation will generate more than
            200M random numbers which may crash the python kernel. It may
            be a better idea to split the simulation into a series of smaller
            sims and then join them after.
            """
        )

    # print half-life of theta
    print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)

    if (eps is None) | (elp is None) | (ejp is None):
        rng = Generator(SFC64(seed))

    if eps is None:
        eps = rng.normal(size=(N, sims))
    else:
        N = eps.shape[0]
        sims = eps.shape[1]
    if elp is None:
        elp = rng.lognormal(mean=_np.log(jump_avgsize), sigma=jump_stdv, size=(N, sims))
    if ejp is None:
        ejp = rng.poisson(lam=jump_prob * dt, size=(N, sims))

    mu = make_into_array(mu, N)
    sigma = make_into_array(sigma, N)

    # repeats first row of eps, elp, and ejp to match the number of simulations
    eps = make_into_array(eps, N).astype(float)
    elp = make_into_array(elp, N).astype(float)
    ejp = make_into_array(ejp, N).astype(float)

    if c == True:
        if len(sigma.shape) == 2:
            sigma = sigma.T.reshape((N + 1) * sims)

        s = _simOUJc(
            s0,
            eps,
            elp,
            ejp,
            theta,
            mu,
            dt,
            sigma,
            sims,
            N,
            mr_lag,
            jump_prob,
            jump_avgsize,
        )
    else:
        # init df with zeros, rows are steps forward in time, columns are simulations
        s = _np.zeros((N + 1, sims))
        s = _pd.DataFrame(data=s)

        # set first row as starting value of sim
        s.iloc[0, :] = s0
        s = _simOUJpy(
            N, s, mu, theta, sigma, jump_prob, jump_avgsize, dt, mr_lag, eps, elp, ejp
        )

    return s


def _simOUJc(
    s0,
    eps,
    elp,
    ejp,
    theta,
    mu,
    dt,
    sigma,
    sims,
    N,
    mr_lag,
    jump_prob,
    jump_avgsize,
):

    # generate a 1D array of random numbers that is based on a
    # 2D array of size P x S where P is the number of time steps
    # including s0 (so N + 1) and S is the number of sims. 1D
    # array will be of size P * S. This is actually the slowest
    # part.
    eps[0, :] = s0

    # make same size as 1D array of all periods and sims
    eps = eps.T.reshape((N + 1) * sims)
    elp = elp.T.reshape((N + 1) * sims)
    ejp = ejp.T.reshape((N + 1) * sims)
    mu = _np.tile(_np.array(mu), sims)
    sigma = _np.tile(_np.array(sigma), sims)

    mr_lag = 0 if mr_lag is None else mr_lag

    x = _csimOUJ(
        x=eps,
        elp=elp,
        ejp=ejp,
        theta=theta,
        mu=mu,
        dt=dt,
        sigma=sigma,
        rows=sims,
        cols=N + 1,
        mr_lag=mr_lag,
        jump_prob=jump_prob,
        jump_avgsize=jump_avgsize,
    )

    return _pd.DataFrame(x.reshape((sims, N + 1)).T)


def _simOUJpy(
    N, s, mu, theta, sigma, jump_prob, jump_avgsize, dt, mr_lag, eps, elp, ejp
):

    mu = _np.vstack([mu] * s.shape[1]).T
    if len(sigma.shape) == 1:
        sigma = _np.vstack([sigma] * s.shape[1]).T

    mu = _pd.DataFrame(mu)
    sigma = _pd.DataFrame(sigma)
    eps = _pd.DataFrame(eps)
    elp = _pd.DataFrame(elp)
    ejp = _pd.DataFrame(ejp)

    for i in range(1, N + 1):
        # calc step
        # fmt: off
        s.iloc[i, :] = (
            s.iloc[i - 1, :]
            + theta
                * (mu.iloc[i, :] - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
            + sigma.iloc[i, :] * s.iloc[i - 1, :] * eps.iloc[i, :] * _np.sqrt(dt)
            + ejp.iloc[i, :] * elp.iloc[i, :]
        )

        if mr_lag is not None:
            # if there is a jump in this step, add it to the mean reversion
            # level so that it doesn't drop back down to the given mean too
            # quickly. Simulates impact of lagged market response to a jump

            mu.iloc[(i):(i + mr_lag), :] += ejp.iloc[i, :] * elp.iloc[i, :]
            ejp.iloc[(i+1):(i + mr_lag), ejp.iloc[i,:] > 0] = 0 # stops double jumps

        # fmt: on

    return s


def fitOU(spread, dt=1 / 252, log_price=False, method="OLS", verbose=False):
    """
    Parameter estimation for the Ornstein-Uhlenbeck process

    Parameters
    ----------
    spread : array-like
        OU process as a list or series to estimate parameters for
    dt : float
        Time step size in fractions of a year. So a day would be 1/252,
        where 252 is the number of business days in a year. Default is 1/252.
        Only used if method is "OLS".
    log_price : bool
        If True, the spread is assumed to be log prices and the log of the spread is taken.
        Default is False.
    method : ['OLS', 'MLE']
        Method to use for parameter estimation. Default is 'OLS'.
    verbose : bool
        If True, prints the estimated parameters. Only used is method is OLS.
        Default is False.

    Returns
    -------
    Dictionary of alpha, mu and theta

    Examples
    --------
    >>> import risktools as rt
    >>> spread = rt.simOU(mu=5, theta=0.5, sigma=0.2, T=5, dt=1/252)
    >>> rt.fitOU(spread[0], method='MLE')
    """

    if log_price == True:
        spread = _np.log(spread)

    if method == "MLE":
        return _fitOU_MLE(spread)
    elif method == "OLS":
        return _fitOU_OLS(spread, dt, verbose)


def _fitOU_MLE(spread):
    """
    Parameter estimation for the Ornstein-Uhlenbeck process

    Parameters
    ----------
    spread : array-like
        OU process as a list or series to estimate parameters for

    Returns
    -------
    Dictionary of alpha, mu and theta

    Examples
    --------
    >>> import risktools as rt
    >>> spread = rt.simOU(mu=5, theta=0.5, signma=0.2, T=5, dt=1/250)
    >>> rt.fitOU(spread)
    """
    spread = _np.array(spread)
    n = len(spread)
    delta = 1

    Sx = spread[:-1].sum()
    Sy = spread[1:].sum()
    Sxx = (spread[:-1] ** 2).sum()
    Syy = (spread[1:] ** 2).sum()
    Sxy = (_np.multiply(spread[:-1], spread[1:])).sum()
    mu = (Sy * Sxx - Sx * Sxy) / ((n - 1) * (Sxx - Sxy) - (Sx**2 - Sx * Sy))

    theta = (
        -_np.log(
            (Sxy - mu * Sx - mu * Sy + (n - 1) * mu**2)
            / (Sxx - 2 * mu * Sx + (n - 1) * mu**2)
        )
        / delta
    )
    a = _np.exp(-theta * delta)
    sigmah2 = (
        Syy
        - 2 * a * Sxy
        + a**2 * Sxx
        - 2 * mu * (1 - a) * (Sy - a * Sx)
        + (n - 1) * mu**2 * (1 - a) ** 2
    ) / (n - 1)
    # print((sigmah2) * 2 * theta / (1 - a**2))
    sigma = _np.sqrt((sigmah2) * 2 * theta / (1 - a**2))
    theta = {"theta": theta, "mu": mu, "annualized_sigma": sigma}

    return theta


def _fitOU_OLS(spread, dt, verbose=False):
    """
    Parameter estimation for the Ornstein-Uhlenbeck process

    Parameters
    ----------
    spread : array-like
        OU process as a list or series to estimate parameters for
    dt : float
        Time step size in fractions of a year. So a day would be 1/252,
        where 252 is the number of business days in a year. Default is 1/252.
        Only used if method is "OLS".
    verbose : bool
        If True, prints the estimated parameters. Default is False.

    Returns
    -------
    Dictionary of annualized sigma, mu and theta

    Examples
    --------
    >>> import risktools as rt
    >>> spread = rt.simOU(mu=5, theta=0.5, signma=0.2, T=5, dt=1/250)
    >>> rt.fitOU(spread)
    """

    if isinstance(spread, _pd.DataFrame):
        raise ValueError("Spread must be a series or array-like, not a dataframe")

    df = _pd.DataFrame()

    df["mrg"] = spread
    df["delta"] = df["mrg"].diff().shift(-1)

    df = df.dropna().reset_index()

    mod = _smf.ols(formula="delta ~ mrg", data=df)
    res = mod.fit()

    theta = -res.params.mrg / dt
    mu = res.params.Intercept / theta / dt
    sigma = res.resid.std() / _np.sqrt(dt)

    if verbose == True:
        print(res.summary())
        print(
            "theta: ",
            round(theta, 2),
            "| mu: ",
            round(mu, 2),
            "| annualized_sigma: ",
            round(sigma, 2),
        )

    return {"theta": theta, "mu": mu, "annualized_sigma": sigma}


if __name__ == "__main__":
    import os
    import sys

    print(os.path.dirname(os.path.realpath(__file__)))

    eps = _pd.read_csv("./pytest/data/diffusion.csv", header=None)

    df = simGBM(
        s0=10, mu=0.0, sigma=0.2, r=0.05, T=1, dt=1 / 252, sims=20, eps=eps
    ).round(2)

    act = _pd.read_csv("./pytest/data/simGBM_output.csv")
    act = act.drop("t", axis=1).T.reset_index(drop=True).T.round(2)

    assert df.equals(act), "simGBM RTL eps failed"

    _np.random.seed(123)
    df = (
        simGBM(s0=10, mu=0.0, sigma=0.2, r=0.05, T=1, dt=1 / 252, sims=20)
        .astype("float")
        .round(4)
    )

    act = (
        _pd.read_csv("./pytest/data/simGBM_output_no_eps.csv").astype("float").round(4)
    )
    act = act.T.reset_index(drop=True).T

    assert df.equals(act), "simGBM generated eps failed"


def OU_lastcol(sims=100000, steps=250, T=25, sigma=0.1, mu=1, theta=1, S0=1):
    # https://stackoverflow.com/questions/24973961/vectorizing-an-equation
    dt = T / steps
    c = 1 - theta * dt
    cv = c ** _np.arange(steps)[::-1]

    R = _np.random.normal(theta * mu * dt, sigma, (sims, steps))

    SN = _np.dot(R, cv) + S0 * c**steps

    df = _pd.DataFrame(SN)
    return df


def stochastic_mu(mu, jump_prob, jump_size, dt, lag, N, sims, seed=None):
    """
    Generate a stochastic mu for the OU processes

    Parameters
    ----------
    mu : float
        The mean of the GBM process
    jump_prob : float
        The probability of a jump from a possion distribution
    jump_size : float
        The size of a jump in the mean
    jump_stdv : float
        The standard deviation of the jump size
    dt : float
        The time step size
    lag : int
        How long the mean reversion level will stay at the jump
        level for in time steps.
    N : int
        The number of time steps
    sims : int
        The number of simulations

    seed : int
        The seed for the random number generator. By default, the seed is None.

    Returns
    -------
    A dataframe of stochastic mu values of size N x sims where N is the number of time steps
    and sims is the number of simulations

    Examples
    --------
    >>> import risktools as rt
    >>> rt.stochastic_mu(mu=4, jump_prob=0.1, jump_avgsize=4, jump_stdv=0.5, dt=1/252, N=100, sims=10)
    """
    rng = Generator(SFC64(seed))
    # elp = rng.lognormal(mean=_np.log(jump_avgsize), sigma=jump_stdv, size=(N, sims))
    elp = _np.ones((N, sims)) * jump_size
    ejp = rng.poisson(lam=jump_prob * dt, size=(N, sims))

    ejp = _np.where(ejp == 0, _np.nan, ejp)
    ejp = _pd.DataFrame(ejp).fillna(method="ffill", axis=0, limit=lag).fillna(0).values

    mu_jump = mu * _np.ones((N, sims))
    mu_jump = mu_jump + elp * ejp

    return mu_jump
