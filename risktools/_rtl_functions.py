import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import quandl
from math import sqrt

def ir_df_us(quandl_key=None, ir_sens=0.01):
    """    
    Extracts US Tresury Zero Rates using Quandl
    
    Parameters
    ----------
    quandl_key : str
        Your Quandl key "yourkey" as a string, by default None. Optional as Quandl allows up to 50
        calls per day for free
    ir_sens : float
        Creates plus and minus IR sensitivity scenarios with specified shock value.

    Returns
    -------
    A pandas data frame of zero rates

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us() 
    """
    # Last 30 days
    sdt = pd.Timestamp.now().floor('D') - pd.DateOffset(days=30)
    
    if quandl_key is not None:
        quandl.ApiConfig.api_key = quandl_key

    fedsfund = quandl.get("FED/RIFSPFF_N_D", start_date=sdt).dropna()
    fedsfund['FedsFunds0'] = np.log((1+fedsfund.Value/360)**365)
    fedsfund.drop('Value', axis=1, inplace=True)

    zero_1yr_plus = quandl.get("FED/SVENY")

    zero_tb = quandl.get(["FED/RIFLGFCM01_N_B", "FED/RIFLGFCM03_N_B", "FED/RIFLGFCM06_N_B"], start_date=sdt).dropna()
    zero_tb.columns = zero_tb.columns.str.replace(" - Value","")

    # get most recent full curve (some more recent days will have NA columns)
    x = fedsfund.join(zero_tb).join(zero_1yr_plus).dropna().iloc[-1,:].reset_index()
    x.columns = ['maturity','yield']
    x['yield'] /= 100 
    x['maturity'] = x.maturity.str.extract('(\d+)').astype('int')

    # change maturity numbers to year fraction for first four rows
    x.maturity[1:4] /= 12
    x.maturity[0] = 1/365

    # add new row for today, same yield as tomorrow
    x = pd.concat(
                [pd.DataFrame({'maturity':[0], 'yield':[x['yield'][0]]}), 
                x], 
                ignore_index=True
            )
    
    x['discountfactor'] = np.exp(-x['yield']*x.maturity)
    x['discountfactor_plus'] = np.exp(-(x['yield']+ir_sens)*x.maturity)
    x['discountfactor_minus'] = np.exp(-(x['yield']-ir_sens)*x.maturity)

    return x


def simGBM(s0=10, drift=0, sigma=0.2, T=1, dt=1/12):
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
    periods = T/dt
    s = [s0]*int(periods)

    for i in range(1,int(periods)):
        s[i] = s[i-1] * np.exp(
            (drift - (sigma**2)/2) * dt +
            sigma * np.random.normal(loc=0, scale=1) * sqrt(dt)
        )

    return s
    

def simOU_arr(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1/252, sims=1000):
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
    periods = int(T/dt)

    if isinstance(mu, list):
        assert (len(mu) == (periods-1)), 'Time dependent mu used, but the length of mu is not equal to the number of periods calculated.'

    # init df with zeros, rows are steps forward in time, columns are simulations
    out = np.zeros((periods, sims))
    out = pd.DataFrame(data=out)

    # set first row as starting value of sim
    out.loc[0,:] = s0

    # print half-life of theta
    print('Half-life of theta in days = ', np.log(2)/theta*bdays_in_year)

    if isinstance(mu, list):
        mu = pd.Series(mu)

    for i, _ in out.iterrows():
        if i == 0: continue # skip first row
    
        # calc gaussian vector
        ep = pd.Series(np.random.normal(size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, pd.Series):
            out.iloc[i,:] = out.iloc[i-1,:] + theta*(mu.iloc[i-1] - out.iloc[i-1,:])*dt + sigma*ep*sqrt(dt)
        else:
            out.iloc[i,:] = out.iloc[i-1,:] + theta*(mu - out.iloc[i-1,:])*dt + sigma*ep*sqrt(dt)

    return out


def simOU(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1/252):
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
    s = np.array(simOU_arr(s0, mu, theta, sigma, T, dt, sims=1).iloc[:,0])

    return s


def simOUJ_arr(s0=5, mu=5, theta=0.5, sigma=0.2, jump_prob=0.05, jump_avgsize=3, jump_stdv=0.05, T=1, dt=1/12, sims=1000):
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
    periods = int(T/dt)

    if isinstance(mu, list):
        assert (len(mu) == (periods-1)), 'Time dependent mu used, but the length of mu is not equal to the number of periods calculated.'

    # init df with zeros, rows are steps forward in time, columns are simulations
    s = np.zeros((periods, sims))
    s = pd.DataFrame(data=s)

    # set first row as starting value of sim
    s.loc[0,:] = s0

    # print half-life of theta
    print('Half-life of theta in days = ', np.log(2)/theta*bdays_in_year)

    if isinstance(mu, list):
        mu = pd.Series(mu)

    for i, _ in s.iterrows():
        if i == 0: continue # skip first row
    
        # calc gaussian and poisson vectors
        ep = pd.Series(np.random.normal(size=sims))
        elp = pd.Series(np.random.lognormal(mean=np.log(jump_avgsize), sigma=jump_stdv, size=sims))
        jp = pd.Series(np.random.poisson(lam=jump_prob*dt, size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, pd.Series):
            s.iloc[i,:] = (s.iloc[i-1,:] 
                + theta*(mu.iloc[i-1] - jump_prob*jump_avgsize - s.iloc[i-1,:])*s.iloc[i-1,:]*dt 
                + sigma*s.iloc[i-1,:]*ep*sqrt(dt)
                + jp*elp
                )
        else:
            s.iloc[i,:] = (s.iloc[i-1,:] 
                + theta*(mu - jump_prob*jump_avgsize - s.iloc[i-1,:])*s.iloc[i-1,:]*dt 
                + sigma*s.iloc[i-1,:]*ep*sqrt(dt)
                + jp*elp
                )

    return s


def simOUJ(s0=5, mu=5, theta=0.5, sigma=0.2, jump_prob=0.05, jump_avgsize=3, jump_stdv=0.05, T=1, dt=1/12):
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
    s = np.array(simOUJ_arr(s0, mu, theta, sigma, jump_prob, jump_avgsize, jump_stdv, T, dt, sims=1).iloc[:,0])

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
    Sxx = (spread[:-1]**2).sum()
    Syy = (spread[1:]**2).sum()
    Sxy = (spread[:-1] * spread[1:]).sum()

    mu = (Sy * Sxx - Sx * Sxy)/((n - 1) * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
    theta = -np.log((Sxy - mu * Sx - mu * Sy + (n - 1) * mu**2)/(Sxx - 2 * mu * Sx + (n - 1) * mu**2))/delta
    a = np.exp(-theta * delta)
    sigmah2 = (Syy - 2 * a * Sxy + a**2 * Sxx - 2 * mu * (1 - a) * (Sy - a * Sx) + (n - 1) * mu**2 * (1 - a)**2)/(n - 1)
    print((sigmah2) * 2 * theta/(1 - a**2))
    sigma = sqrt((sigmah2) * 2 * theta/(1 - a**2))
    theta = {'theta':theta, 'mu':mu, 'sigma':sigma}

    return theta


def bond(ytm=0.05, c=0.05, T=1, m=2, output='price'):
    """
    Compute bond price, cash flow table and duration
    
    Parameters
    ----------
    ytm : float
        Yield to Maturity
    c : float
        Coupon rate per annum
    T : float
        Time to maturity in years
    m : float
        Periods per year for coupon payments e.g semi-annual = 2.
    output : string
        "price", "df" or "duration", by default "df"
    
    Returns
    -------
    Price scalar, cash flows data frame and/or duration scalar
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "price")
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "df")
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "duration")
    """
    assert output in ['df','price','duration'], "output not a member of ['df','price','duration']"

    # init df
    df = pd.DataFrame({'t_years':np.arange(1/m,T+1/m,1/m),'cf':[c*100/m]*(T*m)})
    df['t_periods'] = df.t_years*m
    df.loc[df.t_years==T, 'cf'] = c*100/m+100
    df['disc_factor'] = 1/((1+ytm/m)**df['t_periods'])
    df['pv'] = df.cf*df.disc_factor
    price = df.pv.sum()
    df['duration'] = df.pv*df.t_years/price

    if output == 'price':
        ret = price
    elif output == 'df':
        ret = df
    elif output == 'duration':
        ret = df.duration.sum()
    else:
        ret = None

    return ret


def return_cumulative(r, geometric=True):
    """
    Based on the function Return.annualize from the R package PerformanceAnalytics
    by Peter Carl and Brian G. Peterson

    Calculate a compounded (geometric) cumulative return. Based om R's PerformanceAnalytics
     
    This is a useful function for calculating cumulative return over a period of
    time, say a calendar year.  Can produce simple or geometric return.
    
    product of all the individual period returns
     
    \deqn{(1+r_{1})(1+r_{2})(1+r_{3})\ldots(1+r_{n})-1=prod(1+R)-1}{prod(1+R)-1}
    
    Parameters
    ----------
    r : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True
    
    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> spy = data.DataReader("SPY",  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> spy = spy.pct_change()
    >>> return_cumulative(spy['Adj Close'])
    >>> return_cumulative(spy['Adj Close'],geometric=FALSE)
    """
    r = r.dropna()

    if geometric:
        r = r.add(1).cumprod()-1
    else:
        r = r.sum()

    return r


def return_annualized(r, scale=None, geometric=True):
    """
    Based on the function Return.annualize from the R package PerformanceAnalytics
    by Peter Carl and Brian G. Peterson

    Calculate an annualized return for comparing instruments with different
    length history
     
    An average annualized return is convenient for comparing returns.
    
    Annualized returns are useful for comparing two assets.  To do so, you must
    scale your observations to an annual scale by raising the compound return to
    the number of periods in a year, and taking the root to the number of total
    observations:
    \deqn{prod(1+R_{a})^{\frac{scale}{n}}-1=\sqrt[n]{prod(1+R_{a})^{scale}}-1}{prod(1
    + Ra)^(scale/n) - 1}
    
    where scale is the number of periods in a year, and n is the total number of
    periods for which you have observations.
    
    For simple returns (geometric=FALSE), the formula is:
    
    \deqn{\overline{R_{a}} \cdot scale}{mean(R)*scale}
    
    Parameters
    ----------
    r : Series or DataFrame
        pandas Series with a datetime index with freq defined. Freq must be either 'D','W','M','Q' or 'Y'
    scale : int
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq), 
        specify your own scale to use.
    geometric : bool
        utilize geometric chaining (True) or simple/arithmetic chaining (False) to aggregate returns,
        by default True
    
    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    References
    ----------
    Bacon, Carl. \emph{Practical Portfolio Performance Measurement
    and Attribution}. Wiley. 2004. p. 6

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> return_annualized(r=df[('Adj Close','SPY')], geometric=True)
    >>> return_annualized(r=df[('Adj Close','SPY')], geometric=False)
    >>> return_annualized(r=df['Adj Close'], geometric=True)
    >>> return_annualized(r=df['Adj Close'], geometric=False)
    """
    r, scale = _check_ts(r,scale, name='r')
    
    r = r.dropna()
    n = r.shape[0]

    if geometric:    
        res = (r.add(1).cumprod()**(scale/n)-1).iloc[-1]
    else:
        res = r.mean()*scale
    return res


def return_excess(R, Rf=0):
    """
    Calculates the returns of an asset in excess of the given risk free rate
    
    Calculates the returns of an asset in excess of the given "risk free rate"
    for the period.
    
    Ideally, your risk free rate will be for each period you have returns
    observations, but a single average return for the period will work too.
    
    Mean of the period return minus the period risk free rate
    
    \deqn{\overline{(R_{a}-R_{f})}}{mean(Ra-Rf=0)}
    
    OR
    
    mean of the period returns minus a single numeric risk free rate
    
    \deqn{\overline{R_{a}}-R_{f}}{mean(R)-rf}
    
    Note that while we have, in keeping with common academic usage, assumed that
    the second parameter will be a risk free rate, you may also use any other
    timeseries as the second argument.  A common alteration would be to use a
    benchmark to produce excess returns over a specific benchmark, as
    demonstrated in the examples below.

    Parameters
    ----------
    R : Series or DataFrame
        A time series of asset returns
    Rf : float
        risk free rate, in same period as your returns, or as a single
        digit average
    
    References
    ----------
    Bacon, Carl. \emph{Practical Portfolio Performance Measurement
    and Attribution}. Wiley. 2004. p. 47-52
    
    Examples
    --------
    data(managers)
    head(Return.excess(managers[,1,drop=FALSE], managers[,10,drop=FALSE]))
    head(Return.excess(managers[,1,drop=FALSE], .04/12))
    head(Return.excess(managers[,1:6], managers[,10,drop=FALSE]))
    head(Return.excess(managers[,1,drop=FALSE], managers[,8,drop=FALSE]))
    """

    res = R - Rf
    return res


def sd_annualized(x, scale=None, *args):
    """
    calculate a multiperiod or annualized Standard Deviation
     
    Standard Deviation of a set of observations \eqn{R_{a}} is given by:
    
    \deqn{\sigma = variance(R_{a}) , std=\sqrt{\sigma} }{std = sqrt(var(R))}
    
    It should follow that the variance is not a linear function of the number of
    observations.  To determine possible variance over multiple periods, it
    wouldn't make sense to multiply the single-period variance by the total
    number of periods: this could quickly lead to an absurd result where total
    variance (or risk) was greater than 100%.  It follows then that the total
    variance needs to demonstrate a decreasing period-to-period increase as the
    number of periods increases. Put another way, the increase in incremental
    variance per additional period needs to decrease with some relationship to
    the number of periods. The standard accepted practice for doing this is to
    apply the inverse square law. To normalize standard deviation across
    multiple periods, we multiply by the square root of the number of periods we
    wish to calculate over. To annualize standard deviation, we multiply by the
    square root of the number of periods per year.
    
    \deqn{\sqrt{\sigma}\cdot\sqrt{periods}}
    
    Note that any multiperiod or annualized number should be viewed with
    suspicion if the number of observations is small.
     
    Parameters
    ----------

    x : Series or DataFrame
        A time series of asset returns
    scale : int (optional)
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq), 
        specify your own scale to use.
    *args
        any other passthru parameters
    
    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> sd_annualized(x=df[('Adj Close','SPY')])
    >>> sd_annualized(x=df['Adj Close'])
    """
    if (~isinstance(x, pd.DataFrame) & ~isinstance(x, pd.Series)) == True:
        raise ValueError("x must be a pandas Series or DataFrame")

    if isinstance(x.index, pd.DatetimeIndex):
        if scale is None:
            if (x.index.freq == 'D') | (x.index.freq == 'B'):
                scale = 252
            elif x.index.freq == 'W':
                scale = 52
            elif (x.index.freq == 'M') | (x.index.freq == 'MS'):
                scale = 12
            elif (x.index.freq == 'Q') | (x.index.freq == 'QS'):
                scale = 4
            elif (x.index.freq == 'Y') | (x.index.freq == 'YS'):
                scale = 1
            else:
                raise ValueError("parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'")
    else:
        raise ValueError("parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'")
    
    x = x.dropna()
    res = x.std()*sqrt(scale)
    
    return res    


def omega_sharpe_ratio(R, MAR, *args):
    """
    Omega-Sharpe ratio of the return distribution
    
    The Omega-Sharpe ratio is a conversion of the omega ratio to a ranking statistic 
    in familiar form to the Sharpe ratio.
    
    To calculate the Omega-Sharpe ration we subtract the target (or Minimum
    Acceptable Returns (MAR)) return from the portfolio return and we divide
    it by the opposite of the Downside Deviation.
    
    \deqn{OmegaSharpeRatio(R,MAR) = \frac{r_p - r_t}{\sum^n_{t=1}\frac{max(r_t - r_i, 0)}{n}}}{OmegaSharpeRatio(R,MAR) = (Rp - Rt) / -DownsidePotential(R,MAR)}
    
    where \eqn{n} is the number of observations of the entire series
     
    Parameters
    ----------
    R : Series or DataFrame
        A time series of asset returns
    MAR : int or array_like
        Minimum Acceptable Return, in the same periodicity as your
        returns
    *args:
        any other passthru parameters
    
    Returns
    -------

    Examples
    --------
    >>> mar = 0.005
    >>> print(omega_sharpe_ratio(portfolio_bacon[,1], MAR))
    """
    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index")
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same")

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(R, (pd.Series, pd.DataFrame)):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both 
        # cover the same time period
        MAR = np.mean(MAR)

    result = (upside_risk(R, MAR, stat='potential') - downside_deviation(R, MAR, potential=True)) / downside_deviation(R, MAR, potential=True)
    return result


def upside_risk(R, MAR=0, method='full', stat='risk'):
    """
    upside risk, variance and potential of the return distribution
    
    Upside Risk is the similar of semideviation taking the return above the
    Minimum Acceptable Return instead of using the mean return or zero.

    To calculate it, we take the subset of returns that are more than the target
    (or Minimum Acceptable Returns (MAR)) returns and take the differences of
    those to the target.  We sum the squares and divide by the total number of
    returns and return the square root.
    
    \deqn{ UpsideRisk(R , MAR) = \sqrt{\sum^{n}_{t=1}\frac{
    max[(R_{t} - MAR), 0]^2}{n}}}{UpsideRisk(R, MAR) = sqrt(1/n * sum(t=1..n)
    ((max(R(t)-MAR, 0))^2))}
    
    \deqn{ UpsideVariance(R, MAR) = \sum^{n}_{t=1}\frac{max[(R_{t} - MAR), 0]^2} {n}}{UpsideVariance(R, MAR) = 1/n * sum(t=1..n)((max(R(t)-MAR, 0))^2)}
    
    \deqn{UpsidePotential(R, MAR) = \sum^{n}_{t=1}\frac{max[(R_{t} - MAR), 0]} {n}}{DownsidePotential(R, MAR) =  1/n * sum(t=1..n)(max(R(t)-MAR, 0))}
    
    where \eqn{n} is either the number of observations of the entire series or
    the number of observations in the subset of the series falling below the
    MAR.

    Parameters
    ----------
    R : array_like, Series or DataFrame
        An array or time series of asset returns
    MAR : int
        Minimum Acceptable Return, in the same periodicity as your
        returns
    method : str
        Either "full" or "subset", indicating whether to use the
        length of the full series or the length of the subset of the series below
        the MAR as the denominator, defaults to "full"
    stat : str 
        one of "risk", "variance" or "potential" indicating whether
        to return the Upside risk, variance or potential. By default,
        'risk'
    *args :
        any other passthru parameters
    
    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> print(upside_risk(df['Adj Close'], MAR=0))
    >>> print(upside_risk(df[('Adj Close','SPY')], MAR=0))
    >>> print(upside_risk(df['Adj Close'], MAR=0, stat='variance'))
    >>> print(upside_risk(df['Adj Close'], MAR=0, stat='potential'))
    """

    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index")
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same")

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(R, (pd.Series, pd.DataFrame)):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both 
        # cover the same time period
        MAR = np.mean(MAR)

    if method == 'full':
        length = len(R)
    else:
        length = len(r)

    if stat == 'risk':
        result = np.sqrt(r.sub(MAR).pow(2).div(length).sum())
    elif stat == 'variance':
        result = r.sub(MAR).pow(2).div(length).sum()
    else:
        result = r.sub(MAR).div(length).sum()

    return result


def downside_deviation(R, MAR=0, method='full', potential=False):
    """
    Downside deviation, similar to semi deviation, eliminates positive returns
    when calculating risk.  To calculate it, we take the returns that are less
    than the target (or Minimum Acceptable Returns (MAR)) returns and take the
    differences of those to the target.  We sum the squares and divide by the
    total number of returns to get a below-target semi-variance.

    This is also useful for calculating semi-deviation by setting
    MAR = mean(R)

    Parameters
    ----------
    R : array_like, Series or DataFrame
        An array or time series of asset returns
    MAR : int
        Minimum Acceptable Return, in the same periodicity as your
        returns
    method : str
        Either "full" or "subset", indicating whether to use the
        length of the full series or the length of the subset of the series below
        the MAR as the denominator, defaults to "full"
    potential : bool 
        potential if True, calculate downside potential instead, by default False
    *args :
        any other passthru parameters
    
    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> print(downside_deviation(df[('Adj Close','SPY')], MAR=0))
    >>> print(downside_deviation(df['Adj Close'], MAR=0))
    """

    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index")
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(MAR, (pd.Series, pd.DataFrame)):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError("R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same")

    R = R.dropna()
    r = R[R.lt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(R, (pd.Series, pd.DataFrame)):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both 
        # cover the same time period
        MAR = np.mean(MAR)

    if method == 'full':
        length = len(R)
    else:
        length = len(r)

    if potential:
        result = r.mul(-1).add(MAR).div(length).sum()
    else:
        result = np.sqrt(r.mul(-1).add(MAR).pow(2).div(length).sum())
        # result = r.mul(-1).add(MAR).pow(2).div(length).sum().apply(np.sqrt)    

    return result


def _check_ts(R, scale, name='R'):
    """
    Function to check frequency of R, a time series Series or Dataframe

    Parameters
    ----------
    R : Series or DataFrame
        A time series of asset returns
    scale : int (optional)
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq), 
        specify your own scale to use.

    Returns
    -------
    tuple with R as Series or Dataframe and scale as int
    """
    if (~isinstance(R, pd.DataFrame) & ~isinstance(R, pd.Series)) == True:
        raise ValueError(f"{name} must be a pandas Series or DataFrame")

    if isinstance(R.index, pd.DatetimeIndex):
        if scale is None:
            if (R.index.freq == 'D') | (R.index.freq == 'B'):
                scale = 252
            elif R.index.freq == 'W':
                scale = 52
            elif (R.index.freq == 'M') | (R.index.freq == 'MS'):
                scale = 12
            elif (R.index.freq == 'Q') | (R.index.freq == 'QS'):
                scale = 4
            elif (R.index.freq == 'Y') | (R.index.freq == 'YS'):
                scale = 1
            else:
                raise ValueError(f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'")
    else:
        raise ValueError(f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'")

    return R, scale


def sharpe_ratio_annualized(R, Rf=0, scale=None, geometric=True):
    """
    calculate annualized Sharpe Ratio
    
    The Sharpe Ratio is a risk-adjusted measure of return that uses standard
    deviation to represent risk.
    
    The Sharpe ratio is simply the return per unit of risk (represented by
    variance).  The higher the Sharpe ratio, the better the combined performance
    of "risk" and return.
    
    This function annualizes the number based on the scale parameter.
    
    \deqn{\frac{\sqrt[n]{prod(1+R_{a})^{scale}}-1}{\sqrt{scale}\cdot\sqrt{\sigma}}}
    
    Using an annualized Sharpe Ratio is useful for comparison of multiple return
    streams.  The annualized Sharpe ratio is computed by dividing the annualized
    mean monthly excess return by the annualized monthly standard deviation of
    excess return.
    
    William Sharpe now recommends Information Ratio preferentially to the
    original Sharpe Ratio.

    Parameters
    ----------
    R : array_like
        an xts, vector, matrix, data frame, timeSeries or zoo object of
        asset returns
    Rf : float
        risk free rate, in same period as your returns. By default 0
    scale : int
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None
    geometric : bool
        utilize geometric chaining (True) or simple/arithmetic chaining (False) to aggregate returns,
        default True
    
    see also \code{\link{SharpeRatio}} \cr \code{\link{InformationRatio}} \cr
    \code{\link{TrackingError}} \cr \code{\link{ActivePremium}} \cr
    \code{\link{SortinoRatio}}
    
    References
    ----------
    Sharpe, W.F. The Sharpe Ratio,\emph{Journal of Portfolio
    Management},Fall 1994, 49-58.
    
    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')

    data(managers)
    SharpeRatio.annualized(managers[,1,drop=FALSE], Rf=.035/12) 
    SharpeRatio.annualized(managers[,1,drop=FALSE], Rf = managers[,10,drop=FALSE])
    SharpeRatio.annualized(managers[,1:6], Rf=.035/12) 
    SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE])
    SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE],geometric=FALSE)
    """
    R, scale = _check_ts(R, scale)

    xR = return_excess(R, Rf)
    res = return_annualized(xR, scale=scale, geometric=geometric)
    res /= sd_annualized(R, scale=scale)

    return res


def drawdowns(R, geometric=True, *args):
    """
    Function to calculate drawdown levels in a timeseries

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True

    """

    if geometric:
        res = R.add(1).cumprod()
    else:
        res = R.cumsum()+1

    res = res/res.clip(lower=1).cummax()-1

    return res


def find_drawdowns(R, geometric=True, *args):
    """
    Find the drawdowns and drawdown levels in a timeseries.
    
    find_drawdowns() will find the starting period, the ending period, and
    the amount and length of the drawdown.
    
    Often used with sort_drawdowns() to get the largest drawdowns.
    
    drawdowns() will calculate the drawdown levels as percentages, for use
    in \code{\link{chart.Drawdown}}.
    
    Returns an dictionary: \cr 
    \itemize{ 
      \item return depth of drawdown
      \item from starting period 
      \item to ending period \item length length in periods 
    }
    
    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True
    *args
        any pass-through parameters

    References
    ----------
    Bacon, C. \emph{Practical Portfolio Performance Measurement and
    Attribution}. Wiley. 2004. p. 88 \cr
    
    Examples
    --------
    data(edhec)
    findDrawdowns(edhec[,"Funds of Funds", drop=FALSE])
    sortDrawdowns(findDrawdowns(edhec[,"Funds of Funds", drop=FALSE]))
    """
    dd = drawdowns(R, geometric=geometric).dropna()

    # init result dict
    rs = dict()

    for lab, con in dd.iteritems():
        rs[lab] = dict()
        rs[lab]['return'] = np.array([]).astype(float)
        rs[lab]['from'] = np.array([]).astype(int)
        rs[lab]['to'] = np.array([]).astype(int)
        rs[lab]['length'] = np.array([]).astype(int)
        rs[lab]['trough'] = np.array([]).astype(int)

        if con[0] >= 0:
            prior_sign = 1
        else:
            prior_sign = 0

        frm = 0
        to = 0
        dmin = 0
        sofar = con[0]

        for i, r in enumerate(con): #.iteritems():
            if r < 0:
                this_sign = 0
            else:
                this_sign = 1

            if this_sign == prior_sign:
                if r < sofar:
                    sofar = r
                    dmin = i
                to = i+1
            else:
                rs[lab]['return'] = np.append(rs[lab]['return'],sofar)
                rs[lab]['from'] = np.append(rs[lab]['from'],frm)
                rs[lab]['trough'] = np.append(rs[lab]['trough'],dmin)
                rs[lab]['to'] = np.append(rs[lab]['to'],to)

                frm = i
                sofar = r
                to = i+1
                dmin = i
                prior_sign = this_sign

        rs[lab]['return'] = np.append(rs[lab]['return'],sofar)
        rs[lab]['from'] = np.append(rs[lab]['from'],frm)
        rs[lab]['trough'] = np.append(rs[lab]['trough'],dmin)
        rs[lab]['to'] = np.append(rs[lab]['to'],to)

        rs[lab]['length'] = rs[lab]['to'] - rs[lab]['from'] + 1
        rs[lab]['peaktrough'] = rs[lab]['trough'] - rs[lab]['from'] + 1
        rs[lab]['recovery'] = rs[lab]['to'] - rs[lab]['trough']
    return rs




if __name__ == "__main__":
    np.random.seed(42)

    # sp = simOU()
    # print(fitOU(sp))
    # print(sp)

    # print(bond(output='price'))
    # print(bond(output='duration'))
    # print(bond(output='df'))
    
    from pandas_datareader import data, wb
    from datetime import datetime
    df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    df = df.pct_change()
    df = df.asfreq('B')

    # tt = _sr(df['Adj Close'], Rf=0, scale=252)
    # print(tt)

    # print(drawdowns(df['Adj Close']))
    # print(drawdowns(df[('Adj Close','SPY')]))

    rs = find_drawdowns(df['Adj Close'])
    print(rs['SPY']['return'])
    # print(find_drawdowns(df[('Adj Close','SPY')]))

    # print(sharpe_ratio_annualized(df['Adj Close']))
    # print(sharpe_ratio_annualized(df[('Adj Close','SPY')]))
    
    # print(omega_sharpe_ratio(df['Adj Close'],MAR=0))
    # print(upside_risk(df['Adj Close'], MAR=0))
    # print(upside_risk(df[('Adj Close','SPY')], MAR=0))
    # print(upside_risk(df['Adj Close'], MAR=0, stat='variance'))
    # print(upside_risk(df['Adj Close'], MAR=0, stat='potential'))

    # print(downside_deviation(df[('Adj Close','SPY')], MAR=0))
    # print(downside_deviation(df['Adj Close'], MAR=0))
    
    # print(return_cumulative(r=df['Adj Close'], geometric=True))
    # print(return_cumulative(r=df['Adj Close'], geometric=False))

    # print(return_annualized(r=df[('Adj Close','SPY')], geometric=True))
    # print(return_annualized(r=df[('Adj Close','SPY')], geometric=False))

    # print(return_annualized(r=df['Adj Close'], geometric=True))
    # print(return_annualized(r=df['Adj Close'], geometric=False))

    # print(sd_annualized(x=df[('Adj Close','SPY')]))
    # print(sd_annualized(x=df['Adj Close']))

    ####################
    # R code for testing
    # library(timetk)
    # library(tidyverse)
    # library(PerformanceAnalytics)
    # df <- tq_get('SPY', from='2000-01-01', to='2012-01-01')
    # df <- df %>% dplyr::mutate(adjusted = adjusted/lag(adjusted)-1)
    # R = tk_xts(df, select=adjusted, date_var=date)
    # DownsideDeviation(R, MAR=0)
    # UpsideRisk(R, 0)
    # OmegaSharpeRatio(R, 0)
