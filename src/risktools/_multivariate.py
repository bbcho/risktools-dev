# multivariate simulations

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import plotly.graph_objects as _go


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
    >>> rt.simGBM_MV([100, 100], 0.05, [0.2, 0.3], 1, 0.01, cor=[[1, 0.5], [0.5, 1]], sims=10)
    """
    N = int(T / dt)

    if ~isinstance(s0, _np.ndarray):
        s0 = _np.array(s0)
    if ~isinstance(sigma, _np.ndarray):
        sigma = _np.array(sigma)
    if mu is not None:
        if ~isinstance(mu, _np.ndarray):
            mu = _np.array(mu)
    if ~isinstance(r, _np.ndarray):
        r = _np.array(r)
    if ~isinstance(cor, _np.matrix):
        cor = _np.matrix(cor)

    if mu is None:
        mu = _np.zeros(len(s0))

    sd = _np.diag(sigma)

    if eps is None:
        cov = sd @ cor @ sd
        eps = _np.random.multivariate_normal(mu, cov, size=(N, sims))

    s = _np.zeros((N + 1, sims, len(s0)))
    s[1:, :, :] = _np.exp((r - 0.5 * sigma ** 2) * dt + sigma * _np.sqrt(dt) * eps)
    s[0, :, :] = s0

    return s.cumprod(axis=0)
