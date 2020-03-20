import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.pandas2ri import rpy2py, py2rpy
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

import pandas as pd
import numpy as np
# pandas2ri.activate() # No longer needed in rpy3.0+
# numpy2ri.activate() # No longer needed in rpy3.0+

# R vector of strings
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

tq = importr('tidyquant')
tv = importr('tidyverse')
ql = importr('Quandl')
rtl = importr('RTL')

from rpy2.robjects.packages import STAP
string = """
npv.at.risk <- function(init.cost, C.cost, cf.freq, F, T, disc.factors, simC, X) {

    df <- tibble(t = seq(from = 0, to = T, by = cf.freq), cf = simC) %>% 
      dplyr::mutate(cf = case_when(cf >= X  ~ (cf - C.cost),cf < X ~ 0),
                    cf = replace(cf, t == 0, init.cost), cf = replace(cf, t == T, last(simC)+F),
                    df = spline(x = disc.factors$maturity, y = disc.factors$discountfactor, xout = t)$y, 
                    pv = cf * df)
    
  x = list(df = df, npv = sum(df$pv))
  return(x)
}
"""
_npv_at_risk = STAP(string, "npv.at.risk")

def p2r(p_df):
    # Function to convert pandas dataframes to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(p_df)

    return r_from_pd_df

def r2p(r_df):
    # Function to convert R dataframes to pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(r_df)

    return pd_from_r_df

def ir_df_us(quandlkey=None, ir_sens=0.01):
    """    
    Extracts US Tresury Zero Rates
    
    Parameters
    ----------
    quandlkey : Your Quandl key "yourkey" as a string
    ir_sens : Creates plus and minus IR sensitivity scenarios with specified shock value.

    Returns
    -------
    A pandas data frame of zero rates

    Examples
    --------
    >>> ir = pyRTL.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    """
    
    if quandlkey is not None:
        ir = rtl.ir_df_us(quandlkey, ir_sens)
        return(r2p(ir))
    else:
        print('No Quandl key provided')

def simGBM(S0, drift, sigma, T, dt):
    """
    Simulates a Geometric Brownian Motion process
    
    Parameters
    ----------
    S0 : spot price at time = 0
    drift : drift %
    sigma : standard deviation
    T : maturity in years
    dt : time step in period e.g. 1/250 = 1 business day
    
    Returns
    -------
    A list of simulated values
    
    Examples
    --------
    >>> pyRTL.simGBM(S0=5, drift=0, sigma=0.2, T=2, dt=0.25)
    """
    
    sim = rtl.simGBM(S0, drift, sigma, T, dt)
    
    return(np.array(sim))

def simOU(S0=5,mu=5,theta=.5,sigma=0.2,T=1,dt=1/250):
    """
    Simulates a Ornstein–Uhlenbeck process

    Parameters
    ----------
    S0 : S at time 0
    mu : Mean reversion level
    theta : Mean reversion speed
    sigma : Standard deviation
    T : Maturity in years
    dt : Time step size e.g. 1/250 = 1 business day.

    Returns 
    -------
    A numeric vector of simulated values

    Examples
    --------
    >>> pyRTL.simOU(S0=5,mu=5,theta=.5,sigma=0.2,T=1,dt=1/12)
    """
    
    sim = rtl.simOU(S0,mu,theta,sigma,T,dt)
    return(np.array(sim))

def simOUJ(S0=5,mu=5,theta=10,sigma=0.2,jump_prob=0.05,jump_avesize = 2,jump_stdv = 0.05,T=1,dt=1/250):
    """
    Simulates a Ornstein–Uhlenbeck process with Jumps

    Parameters
    ----------

    S0 : S at t=0
    mu : Mean reversion level
    theta : Mean reversion speed
    sigma : Standard deviation
    jump_prob : Probability of jumps
    jump_avesize : Average size of jumps
    jump_stdv : Standard deviation of jump average size
    T : Maturity in years
    dt : Time step size e.g. 1/250 = 1 business day.

    Returns
    -------
    A numeric list of simulated values

    Examples
    --------
    >>> pyRTL.simOUJ(S0=5,mu=5,theta=.5,sigma=0.2,jump_prob=0.05,jump_avesize = 3,jump_stdv = 0.05,T=1,dt=1/12)
    """
    sim = rtl.simOUJ(S0,mu,theta,sigma,jump_prob,jump_avesize,jump_stdv,T,dt)
    return(np.array(sim))

def fitOU(spread):
    """
    Parameter estimation for Ornstein–Uhlenbeck process

    Parameters
    ----------
    spread : Spread numpy array.

    Returns
    -------
    List of alpha, mu and sigma estimates

    Examples
    --------
    >>> spread = simOU(mu=5,theta=.5,sigma=0.2,T=5,dt=1/250)
    >>> fitOU(spread)
    """

    out = rtl.fitOU(FloatVector(spread))
    outDict = dict()
    outDict['theta'] = np.array(out[0])[0]
    outDict['mu'] = np.array(out[1])[0]
    outDict['sigma'] = np.array(out[2])[0]
    
    return(outDict)

def npv(init_cost , C, cf_freq, F, T, disc_factors, BreakEven, BE_yield):
    """
    Compute NPV
    
    Parameters
    ----------
    init_cost : Initial investment cost
    C : Periodic cash flow
    cf_freq : Cash flow frequency in year fraction e.g. quarterly = 0.25
    F : Final terminal value
    T : Final maturity in years
    disc_factors : Data frame of discount factors using ir_df_us() function. 
    BreakEven : True when using a flat discount rate assumption.
    BE_yield : Set the flat IR rate when BeakEven = TRUE.

    Returns
    -------    
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float
    
    Examples
    --------
    >>> ir = pyRTL.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = pyRTL.npv(init.cost=-375,C=50,cf.freq=.5,F=250,T=2,disc_factors=ir,BreakEven=True,BE_yield=.0399)
    >>> myDict['df']
    >>> myDict['npv']
    """
    tf = rtl.npv(init_cost, C, cf_freq, F, T, disc_factors, BreakEven, BE_yield)
    
    myDict = dict()
    
    myDict['df'] = r2p(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)

def npv_at_risk(init_cost , C_cost, cf_freq, F, T, disc_factors, simC, X):
    """
    NPV at risk function. Replacement for NPV function that takes a list of periodic cashflows as an option instead of a single value
    
    Parameters
    ----------
    init_cost : Initial cost of project (typically negative)
    C_cost : period cost or OPEX
    cf_freq : cash flow frequency in fractions of years (or unit of measure for time to maturity T)
    F : terminal value at maturity
    T : time to maturity (typically years)
    disc_factors : discount factors from interest rate curve
    simC : list of cash flows. Must equal number of periods and formated as Python list
    X : Lower cut off threshold
    
    Returns
    -------
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float
    
    Examples
    --------
    >>> ir = pyRTL.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = pyRTL.npv_at_risk(init.cost=-375,C_cost=5,cf.freq=.5,F=250,T=1,disc_factors=ir,simC=[50,50,50], X=5)
    >>> myDict['df']
    >>> myDict['npv']
    """
    simC = np.asarray(simC)
    simC = ro.FloatVector(simC)
    #disc_factors = ro.FloatVector(disc_factors)
    tf = _npv_at_risk.npv_at_risk(init_cost, C_cost, cf_freq, F, T, disc_factors, simC, X)
    
    myDict = dict()
    
    myDict['df'] = r2p(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)

def _get_RT_data():
    rFunc = """
    
    get_data <- function() {
    
        data <- list( 
                    RTL::cancrudeassays,
                    RTL::cancrudeprices,
                    RTL::df_fut,
                    RTL::dflong,
                    RTL::dfwide,
                    RTL::expiry_table,
                    RTL::holidaysOil,
                    RTL::tickers_eia
                    )
        return(data)
    }
    
    """
    _RTL_data = STAP(rFunc, "get_data")
    
    out = _RTL_data.get_data()
    
    outDict = {'cancrudeassays':r2p(out[0]),
               'cancrudeprices':r2p(out[1]),
               'df_fut':r2p(out[2]),
               'dflong':r2p(out[3]),
               'dfwide':r2p(out[4]),
               'expiry_table':r2p(out[5]),
               'holidaysOil':r2p(out[6]),
               'tickers_eia':r2p(out[7])
               }
    
    return(outDict)
    
def trade_stats(df, Rf = 0):
    """
    Function to compute list of risk reward metrics
     
    Parameters
    ----------

    df {pd.Series}: pandas Series or SINGLE column DataFrame of returns
    Rf {float}: Risk-free rate

    Returns
    -------

    List of risk/reward metrics.

    Examples
    --------

    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime

    >>> spy = data.DataReader("SPY",  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> spy = spy.diff()
    >>> trade_stats(df=spy['Adj Close'],Rf=0)
    """

    # R code to convert R dataframes to xts format
    rFunc = """    
    tradeStatsWrapper <- function(df, Rf) {
        library(xts)
        df$Date <- as.Date(as.character(df$Date))
        df$Close <- as.numeric(df$Close)
        df.xts <- xts(df$Close, order.by=df$Date)
        x = RTL::tradeStats(df.xts, Rf)
        return(x)
    }    
    """

    # rename datetime index
    df.index.name = 'Date'
    
    # if df is a series, convert to dataframe. If it's a dataframe, only take first column and rename it close
    if isinstance(df, pd.Series):
        tf = pd.DataFrame()
        tf['Close'] = df
        df = tf.copy()
    elif isinstance(df, pd.DataFrame):
        df = df.iloc[:,0]
        df.columns = ['Close']
    
    # init R code
    fnc = STAP(rFunc, "tradeStatsWrapper")

    # reset index and drop nans
    df = df.reset_index().dropna()
    
    # convert Date to string since the above code converts it back to R datetime. Probably not needed
    df.Date = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    
    # Convert to pandas df to R df
    rdf = p2r(df)
    
    # Run R code
    out = fnc.tradeStatsWrapper(rdf, Rf)

    # create dictionary from function return
    outDict = {'cumReturn':FloatVector(out[0])[0],
            'retAnnual':FloatVector(out[1])[0],
            'sdAnnual':FloatVector(out[2])[0],
            'sharoe':FloatVector(out[3])[0],
            'omega':FloatVector(out[4])[0],
            'percWin':FloatVector(out[5])[0],
            'percInMkt':FloatVector(out[6])[0],
            'ddLength':FloatVector(out[7])[0],
            'ddMax':FloatVector(out[8])[0]
            }
    
    return outDict