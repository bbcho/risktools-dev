import warnings
import pandas as pd
import numpy as np
import quandl
from math import sqrt
import warnings as _warnings
from . import data
import arch
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


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
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
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
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
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
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
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
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
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
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
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

    Returns
    -------
    Nested dictionary with asset name(s) as the top level key(s). Nested below that as another
    dictionary are the following keys and values:
        'return': numpy array of minimum of returns below the risk free rate of return (Rf) for each 
            trough. If returns are positive, array has 0 value
        'from' : array index positiion of beginning of each trough or recovery period corresponding to each 
            element of 'return'
        'trough' : array index position of peak trough period corresponding to each 
            element of 'return'. Returns beginning of recovery periods
        'to' : array index positiion of end of each trough or recovery period corresponding to each 
            element of 'return'
        'length' : length of each trough period corresponding to each element of 'return' as given by
            the difference in to and from index positions
        'peaktotrough' : array index distance from the peak of each trough or recovery period from the
            beginning of each trough or recovery period, corresponding to each element of 'return'
        'recovery' : array index distance from the peak of each trough or recovery period to the
            end of each trough or recovery period, corresponding to each element of 'return'

    References
    ----------
    Bacon, C. \emph{Practical Portfolio Performance Measurement and
    Attribution}. Wiley. 2004. p. 88 \cr
    
    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> rs = find_drawdowns(df['Adj Close'])
    >>> print(rs['SPY']['peaktotrough'])
    """
    dd = drawdowns(R, geometric=geometric).dropna()

    # init result dict
    rs = dict()

    # convert series into dataframe for flexibility
    series_flag = False
    if isinstance(dd, pd.Series):
        dd = pd.DataFrame({'drawdown':dd})
        series_flag = True

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
        rs[lab]['peaktotrough'] = rs[lab]['trough'] - rs[lab]['from'] + 1
        rs[lab]['recovery'] = rs[lab]['to'] - rs[lab]['trough']

        # if original parameter was a series, remove top layer of
        # results dictionary
        if series_flag == True:
            rs = rs['drawdown']

    return rs


def trade_stats(R, Rf=0):
    """
    Compute list of risk reward metrics

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate 
        cummmulative returns on each column
    Rf : float
        risk free rate, in same period as your returns, or as a single
        digit average

    Returns
    -------
    Dictionary with cummulative returns, annual returns, Annualized Sharpe ratio, Omega Sharpe ratio, 
    Win &, % in the market, and drawdown specs

    Examples
    --------
    >>>
    """

    # r = R.dropna() # don't use, messes up freq check
    r = R.copy()

    rs = dict()

    # convert series into dataframe for flexibility
    series_flag = False
    if isinstance(r, pd.Series):
        r = pd.DataFrame({'trade_stats':r})
        series_flag = True

    for lab, con in r.iteritems():
        rs[lab] = dict()
        y = find_drawdowns(con)
        rs[lab]['cum_ret'] = return_cumulative(con, geometric=True)
        rs[lab]['ret_ann'] = return_annualized(con, scale=252)
        rs[lab]['sd_ann'] = sd_annualized(con, scale=252)
        rs[lab]['omega'] = omega_sharpe_ratio(con, MAR=Rf)*sqrt(252)
        rs[lab]['sharpe'] = sharpe_ratio_annualized(con, Rf=Rf)    
        rs[lab]['perc_win'] = con[con>0].shape[0]/con[con != 0].shape[0]
        rs[lab]['perc_in_mkt'] = con[con != 0].shape[0]/con.shape[0]
        rs[lab]['dd_length'] = max(y['length'])
        rs[lab]['dd_max'] = min(y['return'])

    if series_flag == True:
        rs = rs['trade_stats']

    return rs


def returns(df, ret_type="abs", period_return=1, spread=False):
    """
    Computes periodic returns from a dataframe ordered by date
    
    Parameters
    ----------
    df : Series or DataFrame
        If pandas series, series must have a single datetime index or a multi-index in the same of ('asset name','date'). Values must be asset prices
        If pandas dataframe, dataframe must have a single datetime index with asset prices as columns
    ret_type : str
        "abs" for absolute, "rel" for relative, or "log" for log returns. By default "abs"
    period_return : int
        Number of rows over which to compute returns. By default 1
    spread : bool
        True if you want to spread into a wide dataframe. By default False
    
    Returns
    -------
    A pandas series of returns.
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = True)
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
    """
    df = _check_df(df)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({df.name:df})
    elif isinstance(df, pd.DataFrame) == False:
        raise ValueError("df is not a pandas Series or DataFrame")

    index_flag = False
    if (len(df.index.names) == 1):
        # convert single index into multi-index by adding dummy index in position 0 to allow for groupby
        df['dummy'] = 'dummy'
        df = df.set_index('dummy',append=True)
        df = df.swaplevel()
        index_flag = True

    if (spread == True) & (len(df.index.names) == 1):
        raise ValueError("You have set spread to True but only passed a single index, not a multi-index. There is nothing to spread")

    # calc return type
    if ret_type == "abs":
        df = df.groupby(level=0).apply(lambda x: x.diff())
    elif ret_type == 'rel':
        df = df.groupby(level=0).apply(lambda x: x/x.shift(period_return)-1)
    elif ret_type == 'log':
        if (df[df<0].count().sum() > 0):
            warnings.warn("Negative values passed to log returns. You will likely get NaN values using log returns", RuntimeWarning)
        df = df.groupby(level=0).apply(lambda x: np.log(x/x.shift(period_return)))
    else:
        raise ValueError("ret_type is not valid")

    if spread:
        # return spread df
        df = (df.unstack(level=0)
                .droplevel(level=0, axis=1)
                )
        return df.dropna()
    else:
        if index_flag == True:
            # drop dummy index to return single index
            df = df.droplevel(0)
        return df.dropna().iloc[:,0]


def _returns(df, ret_type="abs", period_return=1, spread=False):
    """
    Computes periodic returns from a dataframe ordered by date
    
    Parameters
    ----------
    df : Series or DataFrame
        pandas series or dataframe with a multi-index in the shape ('asset name','date'). Values should be asset prices
    ret_type : str
        "abs" for absolute, "rel" for relative, or "log" for log returns. By default "abs"
    period_return : int
        Number of rows over which to compute returns. By default 1
    spread : bool
        True if you want to spread into a wide dataframe. By default False
    
    Returns
    -------
    A pandas series of returns.
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = True)
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
    """
    df = _check_df(df)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({'value':df})
    elif isinstance(df, pd.DataFrame):
        df = pd.concat([df], keys='value', axis=1)
    else:
        raise ValueError("df is not a pandas Series or DataFrame")

    index_flag = False
    if isinstance(df.index, pd.Index):
        # convert single index into multi-index to allow for groupby
        df['dummy'] = 'dummy'
        df = df.set_index('dummy',append=True)
        df = df.swaplevel()
        index_flag = True

    if (spread == True) & isinstance(df.index, pd.Index):
        raise ValueError("You have set spread to True but only passed a single index, not a multi-index")

    if ret_type == "abs":
        df['returns'] = df.groupby(level=0).value.apply(lambda x: x.diff())
    elif ret_type == 'rel':
        df['returns'] = df.groupby(level=0).value.apply(lambda x: x/x.shift(period_return)-1)
    elif ret_type == 'log':
        if (df.loc[df['value']<0,'value'].count() > 0) & (df.loc[df['value']>0,'value'].count() > 0):
            warnings.warn("value column as both negative and positive values. You will likely get NaN values using log returns", RuntimeWarning)
        df['returns'] = df.groupby(level=0).value.apply(lambda x: np.log(x/x.shift(period_return)))
    else:
        raise ValueError("ret_type is not valid")

    if spread:
        df = (df.drop('value', axis=1)
                .unstack(level=0)
                .droplevel(level=0, axis=1)
                )
        return df.dropna()
    else:
        if index_flag == True:
            # drop dummy index to return single index
            df = df.droplevel(0)
        return df.dropna().returns
    

def roll_adjust(df, commodity_name="cmewti", roll_type="Last_Trade", roll_sch=None, *args):
    """
    Returns a pandas series adjusted for contract roll. The methodology used to adjust returns is to remove the daily returns on 
    the day after expiry and for prices to adjust historical rolling front month contracts by the size of the roll at 
    each expiry. This is conducive to quantitative trading strategies as it reflects the PL of a financial trader. 
    
    Parameters
    ----------
    df : Series
        pandas series with a with a datetime index and values which are asset prices or with a multi-index in the shape ('asset name','date')
    commodity_name : str
        Name of commodity in expiry_table. See example below for values.
    roll_type : str
        Type of contract roll: "Last_Trade" or "First_Notice". By default "Last_Trade"
    roll_sch : Series
        For future capability. Optional
    *args: Other parms to pass to function
    
    Returns
    -------
    Roll-adjusted pandas series object of returns with datetime index or multi-index with the shape ('asset name','date')
    
    Examples 
    --------
    >>> import risktools as rt
    >>> dflong = rt.data.open_data('dflong')
    >>> rt.data.open_data('expiry_table').cmdty.unique() # for list of commodity names
    >>> ret = rt.returns(df=dflong, ret_type="abs", period_return=1, spread=True)
    >>> ret = ret.iloc[:,0] 
    >>> rt.roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade")
    """

    df = _check_df(df)
    
    if isinstance(df, pd.Series):
        df = pd.DataFrame({df.name:df})
    else:
        raise ValueError("df is not a pandas Series")

    if roll_sch is None:
        roll_sch = data.open_data('expiry_table')
        roll_sch = roll_sch[roll_sch.cmdty==commodity_name]
        roll_sch = roll_sch[roll_type]

    df['expiry'] = df.index.isin(roll_sch, level=-1)
    df['expiry'] = df['expiry'].shift()

    df = df[df.expiry==False].drop('expiry', axis=1)

    return df.iloc[:,0]


def garch(df, out='data', scale=None, show_fig=True, forecast_horizon=1, **kwargs):
    """
    Computes annualised Garch(1,0,1) volatilities using arch package.

    Parameters
    ----------
    df : DataFrame
        Wide dataframe with date column and single series (univariate). 
    out : str 
        "plotly" or "matplotlib" to return respective chart types. "data" to return data or "fit" for garch fit output. By default 'data'
    show_fig : bool
        Only used if out is 'matplotlib' or 'plotly'. If True, function will display plots as well as return their respective fig object
    **kwargs
        key-word parameters to pass tp arch_model. if none, sets p=1, o=0, q=1
    
    Returns
    -------
    out='data' returns a pandas df
    out='fit' returns an arch_model object from the package arch
    out='matplotlib' returns matplotlib figure and axes objects and displays the plot
    out='plotly' returns a plotly figure object and displays the plot
    
    Examples
    --------
    >>> import risktools as rt
    >>> dflong = rt.data.open_data('dflong')
    >>> dflong = dflong['CL01']
    >>> df = rt.returns(df=dflong, ret_type="rel", period_return=1)
    >>> df = rt.roll_adjust(df=df, commodity_name="cmewti", roll_type="Last_Trade")
    >>> rt.garch(df, out="data")
    >>> rt.garch(df, out="fit")
    >>> rt.garch(df, out="plotly")
    >>> rt.garch(df, out="matplotlib")
    """
    df = _check_df(df)

    if kwargs is None:
        kwargs = {'p':1, 'o':0, 'q':1}

    df = df - df.mean()

    freq = pd.infer_freq(df.index[0:10])

    if scale is None:
        if (freq == 'B') or (freq == 'D'):
            scale = 252
        elif (freq[0] == 'W'):
            scale = 52
        elif (freq == 'M') or (freq == 'MS'):
            scale = 12
        elif (freq == 'Q') or (freq == 'QS'):
            scale = 4
        elif (freq == 'Y') or (freq == 'YS'):
            scale = 1
        else:
            raise ValueError("Could not infer frequency of timeseries, please provide scale parameter instead")
            

    # a standard GARCH(1,1) model
    garch = arch.arch_model(df, vol='garch', **kwargs)
    garch_fitted = garch.fit()

    # # calc annualized volatility from variance
    yhat = np.sqrt(garch_fitted.forecast(horizon=forecast_horizon, start=0).variance)*sqrt(scale)

    if out == 'data':
        return yhat
    elif out == 'fit':
        return garch_fitted
    elif out == 'plotly':
        import plotly.express as px
        fig = px.line(yhat)
        fig.show()
        return fig
    elif out =='matplotlib':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        yhat.plot(ax=ax)
        fig.show()
        return fig, ax


def _beta(y, x, subset=None):
    """
    Function to perform a linear regression on y = f(x) -> y = beta*x + const and returns
    beta

    Parameters
    ----------
    y : array-like
        dependent variable
    x : array-like
        independent variable
    subset : array-like of type bool
        array/series/list of bool to subset x and Y

    Examples
    --------
    >>> import numpy as np
    >>> import risktools as rt
    >>> x = np.array([1,2,3,4,5,6,7,8])
    >>> y = x + 5
    >>> rt._beta(x,y)
    """
    x = x.dropna()
    y = y.dropna()
        
    if subset is None:
        subset = np.repeat(True, len(x))
    else:
        subset = subset.dropna().astype(bool)

    if ((isinstance(x, (np.ndarray, pd.Series)) == False) 
            & (isinstance(y, (np.ndarray, pd.Series)) == False) 
            & (isinstance(subset, (np.ndarray, pd.Series)) == False)):
        raise ValueError("all arguements of _beta must be pandas Series or numpy arrays")

    # convert to arrays
    x = np.array(x)
    y = np.array(y)
    subset = np.array(subset)

    # subset
    x = x[subset]
    y = y[subset]

    model = LinearRegression()
    model.fit(x.reshape((-1, 1)),y)
    beta = model.coef_

    return beta[0]


def CAPM_beta(Ra, Rb, Rf=0, kind='all'):
    """
    calculate single factor model (CAPM) beta

    The single factor model or CAPM Beta is the beta of an asset to the variance 
    and covariance of an initial portfolio.  Used to determine diversification potential.

    This function uses a linear intercept model to achieve the same results as
    the symbolic model used by \code{\link{BetaCoVariance}}

    \deqn{\beta_{a,b}=\frac{CoV_{a,b}}{\sigma_{b}}=\frac{\sum((R_{a}-\bar{R_{a}})(R_{b}-\bar{R_{b}}))}{\sum(R_{b}-\bar{R_{b}})^{2}}}{beta
    = cov(Ra,Rb)/var(R)}

    Ruppert(2004) reports that this equation will give the estimated slope of
    the linear regression of \eqn{R_{a}}{Ra} on \eqn{R_{b}}{Rb} and that this
    slope can be used to determine the risk premium or excess expected return
    (see Eq. 7.9 and 7.10, p. 230-231).

    kind='bull' and kind='bear' apply the same notion of best fit to positive and
    negative market returns, separately. kind='bull' is a
    regression for only positive market returns, which can be used to understand
    the behavior of the asset or portfolio in positive or 'bull' markets.
    Alternatively, kind='bear' provides the calculation on negative
    market returns.

    The function \code{TimingRatio} may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio, which is calculated as
    \deqn{TimingRatio =\frac{\beta^{+}}{\beta^{-}}}{Timing Ratio = beta+/beta-}
    is best when greater than one in a rising market and less than one in a
    falling market.

    While the classical CAPM has been almost completely discredited by the 
    literature, it is an example of a simple single factor model, 
    comparing an asset to any arbitrary benchmark.
    
    Parameters
    ----------
    Ra : array-like or DataFrame
        Array-like or DataFrame with datetime index of asset returns to be tested vs benchmark 
    Rb : array-like
        Benchmark returns to use to test Ra
    Rf : array-like | float
        risk free rate, in same period as your returns, or as a single
        digit average
    kind : str
        Market type to return, by default 'all'
        'all' : returns beta for all market types
        'bear' : returns beta for bear markets
        'bull' : returns beta for bull markets

    Returns
    -------
    If Ra is array-like, function returns beta as a scalar. If Ra is a DataFrame, it will return 
    a series indexed with asset names from df columns

    References
    ----------
    Sharpe, W.F. Capital Asset Prices: A theory of market
    equilibrium under conditions of risk. \emph{Journal of finance}, vol 19,
    1964, 425-442. 
    Ruppert, David. \emph{Statistics and Finance, an Introduction}. Springer. 2004. 
    Bacon, Carl. \emph{Practical portfolio performance measurement and attribution}. Wiley. 2004. \cr

    Examples
    --------
    >>> import risktools as rt
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["XOM","AAPL","SPY"],  "yahoo", datetime(2010,1,1), datetime(2017,12,31))
    >>> df = df.pct_change()['Adj Close']
    >>> df = df.asfreq('B')
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0, kind='bear'))
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0, kind='bull'))
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0))
    """

    xRa = return_excess(Ra, Rf)
    xRb = return_excess(Rb, Rf)

    if kind == 'bear':
        subset = xRb.lt(0).mask(xRb.isna(),np.nan) # need to include the mask, otherwise lt/gt returns False for NaN
    elif kind == 'bull':
        subset = xRb.gt(0).mask(xRb.isna(),np.nan) # need to include the mask, otherwise lt/gt returns False for NaN
    else:
        subset = None

    rs = xRa.apply(lambda x: _beta(x, xRb, subset), axis=0)

    return rs

    
def timing_ratio(Ra, Rb, Rf=0):
    """
    The function \code{TimingRatio} may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio, which is calculated as
    \deqn{TimingRatio =\frac{\beta^{+}}{\beta^{-}}}{Timing Ratio = beta+/beta-}
    is best when greater than one in a rising market and less than one in a
    falling market.
    
    Parameters
    ----------
    Ra : array-like or DataFrame
        Array-like or DataFrame with datetime index of asset returns to be tested vs benchmark 
    Rb : array-like
        Benchmark returns to use to test Ra
    Rf : array-like | float
        risk free rate, in same period as your returns, or as a single
        digit average
    
    Returns
    -------
    If Ra is array-like, function returns beta as a scalar. If Ra is a DataFrame, it will return 
    a series indexed with asset names from df columns

    Examples
    --------
    >>> import risktools as rt
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["XOM","AAPL","SPY"],  "yahoo", datetime(2010,1,1), datetime(2017,12,31))
    >>> df = df.pct_change()['Adj Close']
    >>> df = df.asfreq('B')
    >>> rt.timing_ratio(df[['XOM','AAPL']], df['SPY'], Rf=0)
    """

    beta_bull = CAPM_beta(Ra, Rb, Rf=Rf, kind='bull')
    beta_bear = CAPM_beta(Ra, Rb, Rf=Rf, kind='bear')

    result = beta_bull/beta_bear

    return result



# def prompt_beta(df, period='all', beta_type='all', output='plotly'):
#     """
#     Returns betas of multiple prices (by using relative returns).
         
#     Parameters
#     ----------
#     df : DataFrame
#         Wide dataframe with datetime index and multiple series columns (multivariate).
#     period : str
#         "all" or numeric period of time in last n periods, by default 'all'
#     beta_type : str
#         "all" "bull" "bear", by default 'all'
#     output : str
#         'betas', 'plotly','stats', by default 'plotly'
    
#     Returns
#     -------
#     chart, df of betas or stats
        
#     Examples
#     --------
#     >>> import risktools as rt
#     >>> dfwide = rt.data.open_data('dfwide')
#     dfwide = rt.data.open_data('dfwide')
#     col_mask = dfwide.columns[dfwide.columns.str.contains('CL')]
#     dfwide = dfwide[col_mask]
#     x = rt.returns(df=dfwide, ret_type="abs", period_return=1)
#     x = rt.rolladjust(df=x,commodityname=["cmewti"],rolltype=["Last.Trade"])
#     rt.prompt_beta(df=x,period="all",betatype="all",output="chart")
#     rt.prompt_beta(df=x,period="all",betatype="all",output="betas")
#     rt.prompt_beta(df=x,period="all",betatype="all",output="stats")
#     """


def _check_df(df):
    # if isinstance(df.index, pd.DatetimeIndex):
    #     # reset index if the df index is a datetime object
    #     df = df.reset_index().copy()

    return df


# def _infer_freq(x):

#     for c in range(0,10):
#         np.random.

if __name__ == "__main__":
    np.random.seed(42)

    # sp = simOU()
    # print(fitOU(sp))
    # print(sp)

    # print(bond(output='price'))
    # print(bond(output='duration'))
    # print(bond(output='df'))

    # dflong = data.open_data('dflong')
    # ret = returns(df=dflong, ret_type="abs", period_return=1, spread=True).iloc[:,0:2] 
    # roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade").head(50)
    
    # from pandas_datareader import data, wb
    # from datetime import datetime
    # df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    # df = df.pct_change()
    # df = df.asfreq('B')

    # R = df[('Adj Close','SPY')]

    # rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
    # rt.returns(df = rt.data.open_data('dflong'), ret_type = "log", period_return = 1, spread = True)

    # print(y)

    # print(_check_ts(R.dropna(), scale=252))

    # print(trade_stats(df[('Adj Close','SPY')]))
    # print(trade_stats(df['Adj Close']))

    # tt = _sr(df['Adj Close'], Rf=0, scale=252)
    # print(tt)

    # print(drawdowns(df['Adj Close']))
    # print(drawdowns(df[('Adj Close','SPY')]))

    # rs = find_drawdowns(df['Adj Close'])
    # print(rs['SPY']['peaktotrough'])
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
