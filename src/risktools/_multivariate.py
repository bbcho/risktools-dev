# multivariate simulations

from ast import arguments
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import plotly.graph_objects as _go
from ._sims import fitOU, simOU
from abc import ABC as _ABC, abstractmethod as _abstractmethod


def calc_spread_MV(df, formulas):
    """
    Calculate a series of spreads for a multivariate stochastic process.

    Parameters
    ----------
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


def fitOU_MV(df, dt, log_price, method="OLS", verbose=False):
    """
    Fit multiple OU processes

    Parameters
    ----------
    df : DataFrame
        DataFrame of multiple OU processes where each process is a column.  Thus
        a matrix of size (p x N) where p is the number of time steps and N is 
        the number of assets.
    dt : float
        Assumed time step for the OU processes. Must be the same for all
        OU processes.
    log_price : bool
        If True, the spread is assumed to be log prices and the log of the spread is taken.
        All price series must be log_prices or all not. Default is False.
    method : ['OLS' | 'MLE'], optional
        Method used to fit OU process. OLE for ordinary least squares and MLE for 
        Maximum Liklihood. By default 'OLS'.
    verbose : bool
        If True, prints the estimated parameters. Only used is method is OLS.
        Default is False.

    Returns
    -------
    DataFrame of the fitted parameters for each OU process. The rows
    correspond to the parameters and the columns correspond to the OU processes.

    Example
    -------
    >>> import risktools as rt
    """

    params = _pd.DataFrame()

    for c in df.columns:
        ret = fitOU(df[c], dt, log_price=log_price, method=method, verbose=verbose)
        params.loc["theta", c] = ret["theta"]
        params.loc["annualized_sigma", c] = ret["annualized_sigma"]
        params.loc["mu", c] = ret["mu"]

    return params


def generate_eps_MV(cor, T, dt, sims=1000, mu=None):
    """
    Generate epsilons from a multivariate normal distribution
    for use in multivariate stochastic simulations

    Parameters
    ----------
    cor : matrix-like[float]
        Correlation matrix of the OU processes. Must be a square matrix of 
        size N x N and positive definite.
    T : float
        Time horizon of the simulation (in years).
    dt : float
        Time step of the simulation (in years).
    sims : int
        Number of simulations. By default 1000.
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

    # if ~isinstance(sigma, _np.ndarray):
    #     sigma = _np.array(sigma)
    if ~isinstance(cor, _np.matrix):
        cor = _np.matrix(cor)
    if mu is not None:
        if ~isinstance(mu, _np.ndarray):
            mu = _np.array(mu)
    else:
        mu = _np.zeros(cor.shape[0])

    # unneeded since we multiple by sigma in simOU/GBM
    # sd = _np.diag(sigma)
    # cov = sd @ cor @ sd
    cov = cor
    eps = _np.random.multivariate_normal(mu, cov, size=(N, sims))

    return eps


def simGBM_MV(s0, r, sigma, T, dt, mu=None, cor=None, eps=None, sims=1000):
    """
    Simulate Geometric Brownian Motion for stochastic processes with
    multiple assets using a multivariate normal distribution.

    Parameters
    ----------
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
        generated using a multivariate normal distribution. Must be a 3-D array of
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
    >>> rt.simGBM_MV(s0=[100,100], r=0.0, sigma=[0.1,0.1], T=1, dt=1/252, cor=[[1,0],[0,1]], sims=100)
    """
    if (cor is None) & (eps is None):
        raise ValueError("correlation matrix cor required if eps not passed")
    

    if ~isinstance(s0, _np.ndarray):
        s0 = _np.array(s0)
    if ~isinstance(sigma, _np.ndarray):
        sigma = _np.array(sigma)
    if ~isinstance(r, _np.ndarray):
        r = _np.array(r)
    if ~isinstance(cor, _np.matrix):
        cor = _np.matrix(cor)

    if mu is not None:
        if ~isinstance(mu, _np.ndarray):
            mu = _np.array(mu)
    else:
        mu = _np.zeros(len(s0))

    N = int(T / dt)

    if eps is None:
        eps = generate_eps_MV(cor, T, dt, sims, mu)

    s = _np.zeros((N + 1, sims, len(s0)))

    for i in range(0, s.shape[2]):
        s[1:, :, i] = _np.exp(
            (r - 0.5 * sigma[i] ** 2) * dt + sigma[i] * _np.sqrt(dt) * eps[:, :, i]
        )
    s[0, :, :] = s0

    return s.cumprod(axis=0)


def simOU_MV(
    s0, mu, theta, T, dt=None, sigma=None, cor=None, eps=None, sims=1000, **kwargs
):
    """
    Simulate Ornstein-Uhlenbeck process for stochastic processes for
    multiple assets using a multivariate normal distribution.

    Parameters
    ----------
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
        Keyword arguments to pass to simOU function.

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
        eps = generate_eps_MV(cor=cor, T=T, dt=dt, sims=sims)
    else:
        dt = T / eps.shape[0]

    N = int(T / dt)

    s = _np.zeros((N + 1, eps.shape[1], eps.shape[2]))

    for i in range(0, eps.shape[2]):
        s[:, :, i] = simOU(s0[i], mu[i], theta[i], T, dt=dt, eps=eps[:, :, i], **kwargs)

    return s


def generate_random_portfolio_weights(number_assets, number_sims=2500):
    """
    Generate a matrix of random portfolio weights based on
    a number of assets using a uniform distribution.

    Parameters
    ----------
    number_assets : int
        Number of assets in the portfolio.
    number_sims : int, optional
        Number of simulations. By default 2500.

    Returns
    -------
    2-D matrix of random portfolio weights. The first dimension corresponds to the
    simulations and the second dimension corresponds to the assets.

    Example
    -------
    >>> import risktools as rt
    >>> rt.generate_random_portfolio_weights(5, 1000)
    """
    weights = _np.random.uniform(size=(number_sims, number_assets))
    weights = _np.multiply(weights.T, 1 / weights.sum(axis=1)).T

    return weights


def calculate_payoffs(df, strike=0, payoff_funcs=None):
    """
    Calculate the payoffs for a series of simulated assets using asset specific payoff functions.

    Parameters
    ----------

    df : array-like[float]
        Array of (m x n x N) floats where m is the number of periods that the assets are simulated
        forward in time, n is the number of simulations run and N is the number of assets.
    strike : float, optional
        Only used if payoff_funcs is None. Strike price of the option. By default None.
    payoff_funcs : array-like[function], optional
        Array of payoff functions. Must be a 1D array of length N. Each function must take a
        single argument (the simulated asset price) and return a single value (the payoff) along 
        the time axis. By default None. If none, the payoff is the max of the asset price and strike price at
        time T.

    Returns
    -------
    Matrix of payoffs for each simulation. The first dimension corresponds to the
    simulations and the second dimension corresponds to the assets.

    Example
    -------
    >>> import risktools as rt
    >>> def payoff(x):
            ret = _np.clip(x, 0, None)
            return ret.sum(axis=0)
    >>> rt.calc_payoffs(df, payoff_funcs=[payoff, payoff])
    """
    if payoff_funcs is None:
        payoffs = _np.clip(df - strike, 0, None)[-1, :, :]
    else:
        if len(payoff_funcs) != df.shape[2]:
            raise ValueError("Must provide a payoff function for each asset.")

        payoffs = _np.zeros((df.shape[1], df.shape[2]))

        for i in range(0, len(payoff_funcs)):
            payoffs[:, i] = payoff_funcs[i](df[:, :, i])

    return payoffs


def simulate_efficient_frontier(assets, weights):
    """
    Generate portfolio expected returns and risk using simulated
    asset prices and randomized weights.

    Parameters
    ----------
    assets : array-like[float]
        Array of (m x N) floats where m is the simulations of the assets and N is the number of assets.
    weights : array-like[float]
        Array of (n x N) floats where n is the number of portfolios and N is the number of assets.

    Returns
    -------
    Matrix of simulated portfolios. The first dimension corresponds to the
    simulations and the second dimension corresponds to the assets.

    Example
    -------
    >>> import risktools as rt
    >>> assets = rt.simGBM_MV(s0=[100,100], r=0.01, sigma=[0.1,0.1], T=1, dt=1/252, cor=[[1,0.5],[0.5,1]])
    >>> assets = rt.calc_payoffs(assets)
    >>> weights = rt.generate_random_portfolio_weights(5, 1000)
    >>> rt.sim_efficient_frontier(assets, weights)
    """

    out = _np.zeros((weights.shape[0], 2))

    for i in range(0, weights.shape[0]):
        ret = assets @ weights[i]

        out[i, 0] = ret.std()  # risk
        out[i, 1] = ret.mean()  # return

    return out


def make_efficient_frontier_table(returns, weights, asset_names=None):
    """
    Produce dataframe with portfolio returns, risk, and asset weights.

    Parameters
    ----------
    returns : array-like[float]
        Numpy array of (n x 2) floats where n in the number of portfolios simulated. First
        columns is the standard deviation or risk associated with the portfolio and the second
        column is the expected return.
    weights : array-like[float]
        Numpy array of (n x N) floats where n in the number of simulated portfolios and N in the
        number of assets in the portfolio.
    asset_names : list[str]
        List of strings to use as asset names in the output dataframe. Should be of length N
        where N is the number of assets in the portfolio.

    Returns
    -------
    Dataframe with the portfolio risk, expected return and asset weights as columns

    Examples
    --------
    >>> import risktools as rt
    >>> assets = rt.simGBM_MV(s0=[100,100], r=0.01, sigma=[0.1,0.1], T=1, dt=1/252, cor=[[1,0.5],[0.5,1]])
    >>> assets = rt.calc_payoffs(assets)
    >>> weights = rt.generate_random_portfolio_weights(5, 1000)
    >>> port = rt.sim_efficient_frontier(assets, weights)
    >>> port = rt.make_efficient_frontier_table(port, weights, asset_names=['A', 'B'])
    """
    out = _pd.DataFrame(_np.zeros((weights.shape[0], weights.shape[1] + 2)))

    out.iloc[:, [0, 1]] = returns
    out.iloc[:, 2:] = weights

    if asset_names is None:
        asset_names = list(range(1, weights.shape[1] + 1))

    out.columns = ["Risk", "Expected Return"] + asset_names

    return out


def plot_efficient_frontier(df):
    """
    Plot the efficient frontier for simulated portfolios stored in df.

    Parameters
    ----------
    df : DataFrame
        Must be a dataframe with the first two columns being the risk and expected return of the
        portfolio and the remaining columns being the weights of the assets in the portfolio.

    Returns
    -------
    Plot of the efficient frontier.

    Examples
    --------
    >>> import risktools as rt
    >>> assets = rt.simGBM_MV(s0=[100,100], r=0.01, sigma=[0.1,0.1], T=1, dt=1/252, cor=[[1,0.5],[0.5,1]])
    >>> assets = rt.calc_payoffs(assets)
    >>> weights = rt.generate_random_portfolio_weights(5, 1000)
    >>> port = rt.sim_efficient_frontier(assets, weights)
    >>> port = rt.make_efficient_frontier_table(port, weights, asset_names=['A', 'B'])
    >>> rt.plot_efficient_frontier(port)
    """

    df = df.copy()
    N = df.shape[1] - 2

    # convert weights columns into a list of strings with the asset names
    # fmt: off
    for i in range(0, N):
        df.iloc[:, i + 2] = (
            list(df.columns.astype(str))[i + 2] + " = " + df.iloc[:, i + 2].round(2).astype(str)
        )
    # fmt: on

    # build labels for portfolio weights
    df["text"] = df.iloc[:, 2:].apply(
        lambda x: "[" + ", ".join(x.dropna().astype(str)) + "]", axis=1
    )

    fig = _go.Figure()

    fig.add_trace(
        _go.Scattergl(
            x=df.iloc[:, 0],
            y=df.iloc[:, 1],
            marker_color=df.iloc[:, 1] / df.iloc[:, 0],
            mode="markers",
            marker_colorscale="Viridis",
            text=df["text"],
            showlegend=False,
            marker_colorbar=dict(
                title="Sharpe Ratio", titleside="right", tickmode="array",
            ),
        )
    )

    fig.update_layout(
        title='Efficient Frontier',
        title_x = 0.5,
        title_xanchor = 'center',
        xaxis_title='Risk',
        yaxis_title='Expected Returns'
    )

    fig.update_traces(hovertemplate=None)

    return fig


def plot_portfolio(df, weights, fig, weight_names=None, label=True):
    """
    Plot a single portfolio on an efficient frontier chart

    Parameters
    ----------
    df : DataFrame[float] | array-like[float]
        Dataframe or array of simulated asset payoffs from the calc_payoffs function.
    weights : array-like[float]
        Array of asset weights for the portfolio to be plotted. Must sum to 1.
    fig : Figure
        Figure object to plot the portfolio on.
    weight_names : list[str]
        List of strings to use as asset names in the output dataframe. Should be of length N
        where N is the number of assets in the portfolio.
    label : bool
        If True, the portfolio marker will be labeled with the asset weights.

    Returns
    -------
    Plot of the efficient frontier with the portfolio added.

    Examples
    --------
    >>> import risktools as rt
    >>> assets = rt.simGBM_MV(s0=[100,100], r=0.01, sigma=[0.1,0.1], T=1, dt=1/252, cor=[[1,0.5],[0.5,1]])
    >>> assets = rt.calc_payoffs(assets)
    >>> weights = rt.generate_random_portfolio_weights(5, 1000)
    >>> port = rt.sim_efficient_frontier(assets, weights)
    >>> port = rt.make_efficient_frontier_table(port, weights, asset_names=['A', 'B'])
    >>> fig = rt.plot_efficient_frontier(port)
    >>> rt.plot_portfolio(assets, [.5, .5], fig)
    """

    if isinstance(df, _pd.DataFrame):
        df = df.to_numpy()

    # calculate portfolio
    df = df @ _np.array(weights)

    # construct portfolio labels
    if weight_names is not None:
        tmp = zip(weight_names, weights)
        weights = [str(j[0]) + " = " + str(round(j[1], 2)) for j in tmp]
    else:
        weights = [str(round(j, 2)) for j in weights]

    text = "[" + ", ".join([str(w) for w in weights]) + "]"

    x = [df.std()]
    y = [df.mean()]

    # add portfolio to efficient frontier chart
    fig.add_trace(
        _go.Scattergl(
            x=x,
            y=y,
            text=text,
            marker_color="red",
            mode="markers",
            legendgroup="hide",
            showlegend=False,
            marker_size=15,
        )
    )

    if label == True:
        fig.add_annotation(
            x=x[0],
            y=y[0],
            text=text,
            showarrow=False,
            yshift=20,
            font_size=12,
            font_color="red",
        )

    return fig


def shift(xs, n=1):
    # shift along first axis only
    e = _np.empty_like(xs)
    if n >= 0:
        e[:n] = _np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = _np.nan
        e[:n] = xs[-n:]
    return e


class _MVSIM(_ABC):
    """
    Abstract base class for multivariate simulation classes for
    calculating the payoffs of a portfolio of assets.
    """
    _frontier = None
    _sims = None
    _prices = None
    _params = None
    _asset_names = None

    @_abstractmethod
    def fit():
        pass

    @_abstractmethod
    def simulate():
        pass

    def plot_efficient_frontier(self, strike=0, payoff_funcs=None, portfolio_sims=5000):
        """
        Plot the efficient frontier based on a specificied payoff function.

        Parameters
        ----------
        strike : float, optional
            Only used if payoff_funcs is None. Strike price of the option. By default 0.
        payoff_funcs : array-like[function], optional
            Array of payoff functions. Must be a 1D array of length N. Each function must take a
            single argument (the simulated asset price) and return a single value (the payoff) along 
            the time axis. By default None. If none, the payoff is the max of the asset price and strike price at
            time T.
        portfolio_sims : int
            Number of random portfolios to simulate for the efficient frontier.

        Returns
        -------
        Plot of the efficient frontier.

        Examples
        --------
        >>> import risktools as rt

        """

        self._payoffs = calculate_payoffs(self._sims, strike, payoff_funcs)

        # calculate efficient frontier
        weights = generate_random_portfolio_weights(len(self._s0), portfolio_sims)
        port = simulate_efficient_frontier(self._payoffs, weights)

        # make dataframe
        port = make_efficient_frontier_table(
            port, weights, asset_names=self._asset_names
        )

        # plot efficient frontier
        self._frontier = plot_efficient_frontier(port)

        return self._frontier

    def plot_portfolio(self, weights, label=True, weight_names=True):
        """
        Plot a single portfolio on an efficient frontier chart

        Parameters
        ----------
        df : DataFrame[float] | array-like[float]
            Dataframe or array of simulated asset payoffs from the calc_payoffs function.
        weights : array-like[float]
            Array of asset weights for the portfolio to be plotted. Must sum to 1.
        fig : Figure
            Figure object to plot the portfolio on.
        label : bool
            If True, the portfolio marker will be labeled with the asset weights.
        weight_names : bool
            Include weight names in label if already defined in object

        Returns
        -------
        Plot of the efficient frontier with the portfolio added.
        """

        if self._frontier is None:
            ValueError('plot_efficient_frontier method must be run prior to this method')

        names = self._asset_names if weight_names else None

        self._frontier = plot_portfolio(self._payoffs, weights, self._frontier, names, label)

        return self._frontier

    @property
    def sims(self):
        """
        Returns a 3-D array of size (p x sims x N) where p is the number of 
        time steps, sims is the number of simulations, and N is the number 
        of assets
        """
        return self._sims

    @property
    def prices(self):
        if self._prices is None:
            return "No time series we given to initialize the objects"
        else:
            return self._prices

    @property
    def parameters(self):
        params = self._params.copy()
        params.loc['s0',:] = self._s0
        return params

    def plot_sim(self, asset_num, sims=100, **kwargs):
        """
        Function to plot a subset of the simulates run for the asset
        specified by asset_num.

        Parameters
        ----------
        asset_num : int
            Column number of the asset to be shown from 0 to N-1 where
            N is the number of assets passed to the object.
        sims : int
            The number of simulations to show in the chart. Recommend
            around 100. By default 100.
        kwargs
            parameters to be passed to the pandas dataframe plot method.
        
        Returns
        -------
        matplotlib figure object.
        """

        return _pd.DataFrame(self._sims[:,:,asset_num]).iloc[:,:sims].plot(legend=False, **kwargs)


class MVGBM(_MVSIM):
    """
    Class for simulating a multivariate GBM process for a portfolio of assets

    Parameters
    ----------
    r : array-like[float]
        Risk free rate of each asset in the portfolio.
    T : float
        Time to maturity of the portfolio.
    dt : float
        Time step for the simulation.
    s0 : array-like[float], optional
        Initial price of each asset in the portfolio. 
        Required if prices not passed.
    sigma : array-like[float], optional
        Volatility of each asset in the portfolio.
        Required if prices not passed.
    cor : array-like[float], optional
        Correlation matrix for the assets in the portfolio.
        Required if prices not passed.
    prices : array-like[float], optional
        Matrix or dataframe of asset prices where the columns are
        different assets. Used to calibrate model using fit
        method if s0, sigma and cor not passed.
    asset_names : list[str], optional
        List of strings to use as asset names in the output dataframe.
    
    Example
    -------
    >>> import risktools as rt
    >>> mvgbm = rt.MVGBM(
            s0=[100,100, 100], 
            r=0.01, 
            sigma=[0.1,0.1,0.1], 
            T=1, 
            dt=1/252, 
            cor=[[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]],
            asset_names=['A','B','C']
        )
    >>> mvgbm.fit()
    >>> mvgbm.simulate()
    >>> mvgbm.plot_efficient_frontier()
    """

    def __init__(
        self, r, T, dt, s0=None, sigma=None, cor=None, prices=None, asset_names=None
    ):
        self._r = r
        self._T = T
        self._dt = dt
        self._prices = prices
        self._asset_names = asset_names
        self._s0 = s0
        self._sigma = sigma
        self._cor = cor

        if prices is None:
            items = [s0, sigma, cor]
            if any([i is None for i in items]):
                raise ValueError(
                    "Must pass an array of prices to the constructor or provide optional \
                    arguments s0, sigma and cor."
                )

    def fit(self):
        """
        Method to fit the simulation parameters to the class. Not really needed for 
        a GBM simulation, but included for consistency with other simulation classes.
        """
        # function not needed for this class since
        # all parameters are passed in the constructor
        # and there is nothing to fit.

        if self._prices is not None:
            prices = _pd.DataFrame(self._prices)
            returns = (_np.log(prices) - _np.log(prices.shift())).dropna()
            self._s0 = prices.iloc[-1, :]
            self._sigma = returns.std() * _np.sqrt(1 / self._dt)
            self._cor = returns.corr()

    def simulate(self, sims=1000):

        self._sims = simGBM_MV(
            self._s0, self._r, self._sigma, self._T, self._dt, cor=self._cor, sims=sims
        )


class MVOU(_MVSIM):
    """
    Class for simulating a multivariate OU process for a portfolio of assets

    Parameters
    ----------
    s0 : array-like[float], optional
        Initial price of each asset in the portfolio.
        Required if prices not passed.
    mu : float, int or pandas Series, optional
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods).
        Required if prices not passed.
    theta : float, optional
        Mean reversion rate, higher number means it will revert slower.
        Required if prices not passed.
    sigma : array-like[float], optional
        Volatility of each asset in the portfolio.
        Required if prices not passed.
    T : float
        Time to maturity of the portfolio.
    dt : float
        Time step for the simulation.
    cor : array-like[float]
        Correlation matrix for the assets in the portfolio.
    prices : array-like[float], optional
        Matrix or dataframe of asset prices of size (p x N) where p is the number
        of time steps and N is the number of assets. Used to calibrate model using fit
        method if s0, mu, theta, sigma and cor not passed.
    asset_names : list[str]
        List of strings to use as asset names in the output dataframe.
    
    Example
    -------
    >>> import risktools as rt
    >>> ou = rt.MVOU(
            s0=[5,5,5],
            mu=[4,4,4],
            theta=[2,2,2],
            sigma=[0.1,0.1,0.1], 
            T=1, 
            dt=1/252, 
            cor=[[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]],
            asset_names=['A','B','C']
        )
    >>> ou.fit()
    >>> ou.simulate()
    >>> ou.plot_efficient_frontier()
    """
    def __init__(
        self, T, dt, s0=None, mu=None, theta=None, sigma=None, cor=None, prices=None, asset_names=None
    ):
        self._mu = mu
        self._theta = theta
        self._T = T
        self._dt = dt
        self._prices = prices
        self._asset_names = asset_names
        self._s0 = s0
        self._sigma = sigma
        self._cor = cor

        if prices is None:
            items = [s0, mu, theta, sigma, cor]
            if any([i is None for i in items]):
                raise ValueError(
                    "Must pass an array of prices to the constructor or provide optional \
                    arguments s0, mu, theta, sigma and cor."
                )

    def fit(self, log_price=False, method="OLS", verbose=False, s0=None):
        """
        Method to fit the simulation parameters to the class.

        Parameters
        ----------
        log_price : bool
            If True, the spread is assumed to be log prices and the log of the spread is taken.
            Default is False.
        method : ['OLS', 'MLE']
            Method to use for parameter estimation. Default is 'OLS'.
        verbose : bool
            If True, prints the estimated parameters. Only used is method is OLS.
            Default is False.
        s0 : array-like[float]
            Of size N (number of assets) to override the last value for asset prices
            of the passed prices dataframe in the object initializations. Mostly for
            testing and teaching purposes. By default None.
        """
        # Only relevent if prices not passed to constructor
        if self._prices is not None:
            prices = _pd.DataFrame(self._prices)
            returns = (_np.log(prices) - _np.log(prices.shift())).dropna()
            
            self._s0 = prices.iloc[-1, :] if s0 is None else s0

            self._params = fitOU_MV(prices, self._dt, log_price=log_price, method=method, verbose=verbose)
            self._cor = returns.corr()
        else:
            self._params = _pd.DataFrame()
            self._params['mu'] = self._mu
            self._params['annualized_sigma'] = self._sigma
            self._params['theta'] = self._theta
            self._params = self._params.T

        if self._asset_names is not None:
            self._params.columns = self._asset_names

    def simulate(self, sims=1000):

        self._sims = self._sims = simOU_MV(
            self._s0, self._params.loc['mu',:], self._params.loc['theta',:], self._T, self._dt, self._params.loc['annualized_sigma',:], 
            self._cor, sims=sims
        )

