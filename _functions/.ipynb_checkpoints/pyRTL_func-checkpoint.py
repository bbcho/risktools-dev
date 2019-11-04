import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.pandas2ri import ri2py, py2ri
from rpy2.robjects.packages import importr

import pandas as pd
import numpy as np
pandas2ri.activate()
numpy2ri.activate()

# R vector of strings
from rpy2.robjects.vectors import StrVector

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

tq = importr('tidyquant')
tv = importr('tidyverse')
ql = importr('Quandl')
rtl = importr('RTL')

def ir_df_us(quandlkey=None, ir_sens=0.01) -> pd.DataFrame():
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
        return(ri2py(ir))
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

def simOU(S0=5,mu=5,theta=.5,sigma=0.2,T=1,dt=1/12):
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
    spread : Spread pandas Series.

    Returns
    -------
    List of alpha, mu and sigma estimates

    Examples
    --------
    >>> spread = rtl.simOU(mu=5,theta=.5,sigma=0.2,T=5,dt=1/250)
    >>> rtl.fitOU(spread)
    """

    out = rtl.fitOU(spread)
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
    
    myDict['df'] = ri2py(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)


from rpy2.robjects.packages import STAP

# with open('./pyRTL/_functions/npvAtRisk.R', 'r') as f:
#     string = f.read()

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
    >>> myDict = pyRTL.npv(init.cost=-375,C_cost=5,cf.freq=.5,F=250,T=1,disc_factors=ir,simC=[50,50,50], X=5)
    >>> myDict['df']
    >>> myDict['npv']
    """
    simC = np.asarray(simC)
    simC = ro.FloatVector(simC)
    #disc_factors = ro.FloatVector(disc_factors)
    tf = _npv_at_risk.npv_at_risk(init_cost, C_cost, cf_freq, F, T, disc_factors, simC, X)
    
    myDict = dict()
    
    myDict['df'] = ri2py(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)


