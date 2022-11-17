import pandas as _pd
import numpy as _np
import statsmodels.formula.api as _smf


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
        eps = _np.random.normal(mu, sigma, size=(periods, sims))

    s.loc[0, :] = s0
    s.loc[1:, :] = eps

    # calc geometric brownian motion
    s.loc[1:, :] = _np.exp(
        (r - sigma ** 2 / 2) * dt + sigma * _np.sqrt(dt) * s.loc[1:, :]
    )

    s = s.cumprod()

    return s


def simOU(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252, sims=1000, eps=None):
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
        Number of simulations to run. By default, this is 1000.
    eps : matrix-like[float]
        Random numbers to use for the returns. If provided, mu, sigma, T, dt and sims are ignored.
        Must of size (p x sims) where p is the number of periods in T/dt.

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    periods = int(T / dt)

    if isinstance(mu, list):
        assert len(mu) == (
            periods
        ), "Time dependent mu used, but the length of mu is not equal to the number of periods calculated."

    # init df with zeros, rows are steps forward in time, columns are simulations
    out = _np.zeros((periods + 1, sims))
    out = _pd.DataFrame(data=out)

    # set first row as starting value of sim
    out.loc[0, :] = s0

    # print half-life of theta
    print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)

    if isinstance(mu, list):
        mu = _pd.Series(mu)

    # calc gaussian vector
    if eps is None:
        eps = _np.random.normal(size=(periods, sims))

    for i, _ in out.iterrows():
        if i == 0:
            continue  # skip first row

        # calc step
        if isinstance(mu, list) | isinstance(mu, _pd.Series):
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu.iloc[i - 1] - out.iloc[i - 1, :]) * dt
                + sigma * eps[i - 1, :] * _np.sqrt(dt)
            )
        else:
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu - out.iloc[i - 1, :]) * dt
                + sigma * eps[i - 1, :] * _np.sqrt(dt)
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
    mr_lag : int
        Lag in mean reversion. If None, then no lag is used. If > 0, then the diffusion does not immediately
        return the mean after a jump at theta but instead with remain near the jump level for mr_lag periods.
        By default, this is None.

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
    else:
        # turn scalar mu into a series to iterate over
        mu = _pd.Series(_np.ones(periods) * mu)

    # turn mu series into a dataframe to iterate over
    mu = _pd.concat([mu] * sims, axis=1)

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
        # fmt: off

        s.iloc[i, :] = (
            s.iloc[i - 1, :]
            + theta
                * (mu.iloc[i - 1, :] - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
            + sigma * s.iloc[i - 1, :] * ep * _np.sqrt(dt)
            + jp * elp
        )

        if mr_lag is not None:
            # if there is a jump in this step, add it to the mean reversion
            # level so that it doesn't drop back down to the given mean too
            # quickly. Simulates impact of lagged market response to a jump
            mu.iloc[(i):(i + mr_lag), :] += jp * elp

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
    import risktools as rt

    sim = rt.simOU(s0=5, mu=4, theta=2, sigma=0.2, T=5, dt=1 / 252)

    params = rt.fitOU(sim, dt=1 / 252, method="OLS", verbose=True)

    print(params)

    params = rt.fitOU(sim, dt=1 / 252, method="MLE", verbose=True)

    print(params)
