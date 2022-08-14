# multivariate simulations

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import plotly.graph_objects as _go
from ._sims import fitOU, simOU


def calc_spread_MV(df, formulas):
    """
    Calculate a series of spreads for a multivariate stochastic process.

    df : DataFrame
        DataFrame containing the simulated values of the stochastic processes.
        The columns correspond to the assets and the index corresponds to the
        time steps.
    formulas : dictionary
        Dictionary of formulas to use for calculating the spreads. The key must be
        the name of the spread and the value must be a string containing the formula
        for calculating the spread. The formula must be a valid Python expression using
        the names of the columns in df as variables. For example, to calculate the spread 
        between asset_1 and asset_2 less 5, the formula would be 'asset_1 - asset_2 - 5'.

    Returns
    -------

    DataFrame containing the simulated values of the spreads. The columns
    correspond to the spreads and the index corresponds to the time steps.

    Example
    -------
    >>> import risktools as rt
    >>> df = rt.simGBM_MV([100, 100], 0.05, [0.2, 0.3], 1, 0.01, cor=[[1, 0.5], [0.5, 1]], sims=10)
    >>> rt.calc_spread_MV(df, {'spread':'1-2'})
    """
    spreads = _pd.DataFrame(index=df.index)

    for i, r in enumerate(formulas.keys()):
        spreads[i] = df.eval(formulas[r])

    spreads.columns = list(formulas.keys())

    return spreads


def fitOU_MV(df, dt, method="OLS"):
    """
    Fit multiple OU processes

    df : DataFrame
        DataFrame of multiple OU processes where each process is a column
    dt : float
        Assumed time step for the OU processes. Must be the same for all
        OU processes.
    method : ['OLS' | 'MLE'], optional
        Method used to fit OU process. OLE for ordinary least squares and MLE for 
        Maximum Liklihood. By default 'OLS'.

    Returns
    -------

    DataFrame of the fitted parameters for each OU process. The columns
    correspond to the parameters and the rows correspond to the OU processes.

    Example
    -------
    >>> import risktools as rt
    """

    params = _pd.DataFrame()

    for c in df.columns:
        ret = fitOU(df[c], dt)
        params.loc["theta", c] = ret["theta"]
        params.loc["annualized_sigma", c] = ret["annualized_sigma"]
        params.loc["mu", c] = ret["mu"]

    return params


def generate_eps_MV(sigma, cor, T, dt, sims, mu=None):
    """
    Generate multivariate epsilons for use in multivariate stochastic simulations

    sigma : array-like[float]
        Array of annualized standard deviations to use for each OU or GBM process
    cor : matrix-like[float]
        Correlation matrix of the OU processes. Must be a square matrix of 
        size N x N and positive definite.
    T : float
        Time horizon of the simulation (in years).
    dt : float
        Time step of the simulation (in years).
    sims : int
        Number of simulations.
    mu : array-like[float], optional
        Array of means to use for the multivariate normal for each random process. 
        If None, mu = 0 is used for all random processes. By default None.

    Returns
    -------

    Matrix of random numbers to use for the simulation. The first dimension
    corresponds to the time steps, the second dimension corresponds to the simulations,
    and the third dimension corresponds to the OU processes.

    Example
    -------
    >>> import risktools as rt
    >>> rt.generate_eps_MV([0.2, 0.3], [[1, 0.5], [0.5, 1]], 1, 0.01, 10)
    """
    N = int(T / dt)

    if ~isinstance(sigma, _np.ndarray):
        sd = _np.array(sd)
    if ~isinstance(cor, _np.matrix):
        cor = _np.matrix(cor)
    if mu is not None:
        if ~isinstance(mu, _np.ndarray):
            mu = _np.array(mu)
    else:
        mu = _np.zeros(len(sd))

    sd = _np.diag(sigma)

    cov = sd @ cor @ sd
    eps = _np.random.multivariate_normal(mu, cov, size=(N, sims))

    return eps


def simGBM_MV(s0, r, sigma, T, dt, mu=None, cor=None, eps=None, sims=1000):
    """
    Simulate Geometric Brownian Motion for stochastic processes with
    multiple assets using a multivariate normal distribution.

    s0 : array-like
        Initial values of the stochastic processes. Must be a 1D array of length
        N where N is the number of assets.
    r : float
        Risk-free rate.
    sigma : array-like
        Volatility of the stochastic processes (annualized standard deviations of 
        returns). Must be a 1D array of length N. Only used if eps is None.
    T : float
        Time horizon of the simulation (in years).
    dt : float
        Time step of the simulation (in years).
    mu : array-like, optional
        Means to use for multivariate normal distribution of returns. Must be
        a 1D array of length N. If None, mu = 0 is used for all assets. Only used
        if eps is None.
    cor : matrix-like
        Correlation matrix of the stochastic processes. Must be a square matrix of 
        size N x N and positive definite. Only used if eps is None.
    eps : array-like, optional
        Random numbers to use for the simulation. If not provided, random numbers are
        generated using a multivariate normal distribution. Must be a 2D array of
        size (p x sims x N) where p is the number of time steps, sims is the number of
        simulations, and N is the number of assets. By default None.
    sims : int
        Number of simulations. By default 1000.

    Returns
    -------

    Matrix of simulated values of the stochastic processes. The first dimension
    corresponds to the time steps, the second dimension corresponds to the simulations,
    and the third dimension corresponds to the assets.

    Example
    -------
    >>> import risktools as rt
    >>> rt.simGBM_MV(s0=[100,100], r=0.0, sigma=[0.1,0.1], T=1, dt=1/252, cor=cor, sims=100)
    """
    N = int(T / dt)

    if ~isinstance(s0, _np.ndarray):
        s0 = _np.array(s0)
    if ~isinstance(sigma, _np.ndarray):
        sigma = _np.array(sigma)
    if mu is not None:
        if ~isinstance(mu, _np.ndarray):
            mu = _np.array(mu)
    else:
        mu = _np.zeros(len(s0))
    if ~isinstance(r, _np.ndarray):
        r = _np.array(r)
    if ~isinstance(cor, _np.matrix):
        cor = _np.matrix(cor)

    if eps is None:
        eps = generate_eps_MV(sigma, cor, T, dt, sims, mu)

    s = _np.zeros((N + 1, sims, len(s0)))
    s[1:, :, :] = _np.exp((r - 0.5 * sigma ** 2) * dt + sigma * _np.sqrt(dt) * eps)
    s[0, :, :] = s0

    return s.cumprod(axis=0)


def simOU_MV(
    s0, mu, theta, T, dt=None, sigma=None, cor=None, eps=None, sims=1000, **kwargs
):
    """
    Simulate Ornstein-Uhlenbeck process for stochastic processes with
    multiple assets using a multivariate normal distribution.

    s0 : array-like[float]
        Initial values of the stochastic processes. Must be a 1D array of length
        N where N is the number of assets.
    mu : array-like[float]
        Mean of the OU processes. Must be a 1D array of length N.
    theta : array-like[float]
        Mean reversion parameter of the OU processes. Must be a 1D array of length N.
    T : float
        Time horizon of the simulation (in years).
    dt : float, optional
        Time step of the simulation (in years). Not used if eps is provided. By default None.
    sigma : array-like[float], optional
        Volatility of the OU processes (annualized standard deviations of
        returns). Must be a 1D array of length N. Only used if eps is None.
    cor : matrix-like[float], optional
        Correlation matrix of the OU processes. Must be a square matrix of
        size N x N and positive definite. Only used if eps is None.
    eps : matrix-like, optional
        Random numbers to use for the simulation. Must be a 2D array of
        size (p x sims x N) where p is the number of time steps, sims is the number of
        simulations, and N is the number of assets. By default None.
    sims : int
        Number of simulations. By default 1000. Not used if eps is provided.
    **kwargs : optional
        Keyword arguments to pass to simOU.

    Returns
    -------

    Matrix of simulated values of the stochastic processes. The first dimension
    corresponds to the time steps, the second dimension corresponds to the simulations,
    and the third dimension corresponds to the assets.

    Example
    -------
    >>> import risktools as rt
    >>> rt.simOU_MV(s0=[100,100], mu=[0.1,0.1], theta=[0.1,0.1], T=1, dt=1/252, eps=eps)
    """

    if eps is None:
        if (T is None) | (dt is None):
            raise ValueError("Must provide T and dt if eps is not provided.")
        eps = generate_eps_MV(sigma=sigma, cor=cor, T=T, dt=dt, sims=sims)
    else:
        dt = T / eps.shape[0]

    N = int(T / dt)

    s = _np.zeros((N + 1, eps.shape[1], eps.shape[2]))

    for i in range(0, eps.shape[2]):
        s[:, :, i] = simOU(s0[i], mu[i], theta[i], T, dt=dt, eps=eps[:, :, i], **kwargs)

    return s
