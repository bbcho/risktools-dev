import pandas as _pd
import numpy as _np

# from math import sqrt


def simGBM(s0=10, drift=0, sigma=0.2, T=1, dt=1 / 12):
    """
    Simulates a Geometric Brownian Motion process

    Parameters
    ----------
    s0 : spot price at time = 0
    drift : drift %
    sigma : standard deviation
    T : maturity in years
    dt : time step in period e.g. 1/250 = 1 business day

    Returns
    -------
    A list of simulated values

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simGBM(s0=5, drift=0, sigma=0.2, T=2, dt=0.25)
    """
    periods = T / dt
    s = [s0] * int(periods)

    for i in range(1, int(periods)):
        s[i] = s[i - 1] * _np.exp(
            (drift - (sigma ** 2) / 2) * dt
            + sigma * _np.random.normal(loc=0, scale=1) * _np.sqrt(dt)
        )

    return s


def simOU_arr(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252, sims=1000):
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
    mu : float, int or pandas Series
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods)
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU_arr()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    periods = int(T / dt)

    if isinstance(mu, list):
        assert len(mu) == (
            periods - 1
        ), "Time dependent mu used, but the length of mu is not equal to the number of periods calculated."

    # init df with zeros, rows are steps forward in time, columns are simulations
    out = _np.zeros((periods, sims))
    out = _pd.DataFrame(data=out)

    # set first row as starting value of sim
    out.loc[0, :] = s0

    # print half-life of theta
    print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)

    if isinstance(mu, list):
        mu = _pd.Series(mu)

    for i, _ in out.iterrows():
        if i == 0:
            continue  # skip first row

        # calc gaussian vector
        ep = _pd.Series(_np.random.normal(size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, _pd.Series):
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu.iloc[i - 1] - out.iloc[i - 1, :]) * dt
                + sigma * ep * _np.sqrt(dt)
            )
        else:
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu - out.iloc[i - 1, :]) * dt
                + sigma * ep * _np.sqrt(dt)
            )

    return out


def simOU(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk)

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
        Mean that the function will revert to
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year

    Returns
    -------
    A numpy array of simulated values

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU()
    """
    s = _np.array(simOU_arr(s0, mu, theta, sigma, T, dt, sims=1).iloc[:, 0])

    return s


def simOUJ_arr(
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
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    jump_prob : float
        Probablity of jumps
    jump_avgsize : float
        Average size of jumps
    jump_stdv : float
        Standard deviation of average jump size
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOUJ_arr()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    periods = int(T / dt)

    if isinstance(mu, list):
        assert len(mu) == (
            periods - 1
        ), "Time dependent mu used, but the length of mu is not equal to the number of periods calculated."

    # init df with zeros, rows are steps forward in time, columns are simulations
    s = _np.zeros((periods, sims))
    s = _pd.DataFrame(data=s)

    # set first row as starting value of sim
    s.loc[0, :] = s0

    # print half-life of theta
    print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)

    if isinstance(mu, list):
        mu = _pd.Series(mu)

    for i, _ in s.iterrows():
        if i == 0:
            continue  # skip first row

        # calc gaussian and poisson vectors
        ep = _pd.Series(_np.random.normal(size=sims))
        elp = _pd.Series(
            _np.random.lognormal(mean=_np.log(jump_avgsize), sigma=jump_stdv, size=sims)
        )
        jp = _pd.Series(_np.random.poisson(lam=jump_prob * dt, size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, _pd.Series):
            s.iloc[i, :] = (
                s.iloc[i - 1, :]
                + theta
                * (mu.iloc[i - 1] - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
                + sigma * s.iloc[i - 1, :] * ep * _np.sqrt(dt)
                + jp * elp
            )
        else:
            s.iloc[i, :] = (
                s.iloc[i - 1, :]
                + theta
                * (mu - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
                + sigma * s.iloc[i - 1, :] * ep * _np.sqrt(dt)
                + jp * elp
            )

    return s


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
):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk) with Jump

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
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    jump_prob : float
        Probablity of jumps
    jump_avgsize : float
        Average size of jumps
    jump_stdv : float
        Standard deviation of average jump size
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOUJ()
    """
    s = _np.array(
        simOUJ_arr(
            s0, mu, theta, sigma, jump_prob, jump_avgsize, jump_stdv, T, dt, sims=1
        ).iloc[:, 0]
    )

    return s


def fitOU(spread):
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
    n = len(spread)
    delta = 1

    Sx = spread[:-1].sum()
    Sy = spread[1:].sum()
    Sxx = (spread[:-1] ** 2).sum()
    Syy = (spread[1:] ** 2).sum()
    Sxy = (spread[:-1] * spread[1:]).sum()

    mu = (Sy * Sxx - Sx * Sxy) / ((n - 1) * (Sxx - Sxy) - (Sx ** 2 - Sx * Sy))
    theta = (
        -_np.log(
            (Sxy - mu * Sx - mu * Sy + (n - 1) * mu ** 2)
            / (Sxx - 2 * mu * Sx + (n - 1) * mu ** 2)
        )
        / delta
    )
    a = _np.exp(-theta * delta)
    sigmah2 = (
        Syy
        - 2 * a * Sxy
        + a ** 2 * Sxx
        - 2 * mu * (1 - a) * (Sy - a * Sx)
        + (n - 1) * mu ** 2 * (1 - a) ** 2
    ) / (n - 1)
    print((sigmah2) * 2 * theta / (1 - a ** 2))
    sigma = _np.sqrt((sigmah2) * 2 * theta / (1 - a ** 2))
    theta = {"theta": theta, "mu": mu, "sigma": sigma}

    return theta
