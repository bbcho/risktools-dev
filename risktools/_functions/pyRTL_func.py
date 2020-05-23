import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.pandas2ri import rpy2py, py2rpy
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
import numpy as np
from rpy2.ipython.ggplot import image_png

from contextlib import contextmanager
from rpy2.robjects.lib import grdevices
from IPython.display import Image, display

import pandas as pd
import numpy as np
# pandas2ri.activate() # No longer needed in rpy3.0+
# numpy2ri.activate() # No longer needed in rpy3.0+

# R vector of strings
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector, Vector

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
    with localconverter(ro.default_converter + pandas2ri.converter) as cv:
        # r_from_pd_df = ro.conversion.py2rpy(p_df)
        r_from_pd_df = cv.py2rpy(p_df)

    return r_from_pd_df

def r2p(r_df):
    # Function to convert R dataframes to pandas
    with localconverter(ro.default_converter + pandas2ri.converter) as cv:
        # pd_from_r_df = ro.conversion.rpy2py(r_df)
        pd_from_r_df = cv.rpy2py(r_df)

    return pd_from_r_df

def convert_usSwapCurves_2p(x):
    """
    Function designed to convert nested list usSwapCurves from R package RTL to python
    
    x {ListVector}
    """
    out = dict()
    out['times'] = np.array(x[0])
    out['discounts'] = np.array(x[1])
    out['forwards'] = np.array(x[2])
    out['zerorates'] = np.array(x[3])
    out['flatQuotes'] = bool(np.array(x[4])[0])
    
    out['params'] = dict()
    
    out['params']['tradeDate'] = pd.to_datetime(x[5][0][0],unit='D', utc=True)
    out['params']['settleDate'] = pd.to_datetime(x[5][1][0],unit='D', utc=True)
    out['params']['dt'] = x[5][2][0]
    out['params']['interpWhat'] = x[5][3][0]
    out['params']['interpHow'] = x[5][4][0]
    out['table'] = r2p(x[6])
    
    out['table']['date'] = pd.to_datetime(out['table']['date'],unit='D', utc=True)
    
    return out

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
                    RTL::tickers_eia,                    
                    RTL::ng_storage,
                    RTL::tickers_eia,
                    RTL::tradeCycle,
                    RTL::twoott,
                    RTL::twtrump,
                    RTL::usSwapCurves,
                    RTL::usSwapCurvesPar,
                    RTL::usSwapIR,
                    RTL::usSwapIRdef
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
               'tickers_eia':r2p(out[7]),
               'ng_storage':r2p(out[8]),
               'tickers_eia':r2p(out[9]),
               'tradeCycle':r2p(out[10]),
               'twoott':r2p(out[11]),
               'twtrump':r2p(out[12]),
               'usSwapCurves':r2p(out[13]),
               'usSwapCurvesPar':r2p(out[14]),
               'usSwapIR':r2p(out[15]),
               'usSwapIRdef':r2p(out[16])
               }
    
    # For some reason, pure dates (i.e. no times), when converted from R to py breaks dates. 
    # Need to convert back to pandas datetime using unit 'days'
    outDict['df_fut'].date = pd.to_datetime(outDict['df_fut'].date,unit='D', utc=True)
    outDict['dflong'].date = pd.to_datetime(outDict['dflong'].date,unit='D', utc=True)
    outDict['dfwide'].date = pd.to_datetime(outDict['dfwide'].date,unit='D', utc=True)
    
    outDict['expiry_table']['Last.Trade'] = pd.to_datetime(outDict['expiry_table']['Last.Trade'], unit='D', utc=True)
    outDict['expiry_table']['First.Notice'] = pd.to_datetime(outDict['expiry_table']['First.Notice'], unit='D', utc=True)
    outDict['expiry_table']['First.Delivery'] = pd.to_datetime(outDict['expiry_table']['First.Delivery'], unit='D', utc=True)
    outDict['expiry_table']['Last.Delivery'] = pd.to_datetime(outDict['expiry_table']['Last.Delivery'], unit='D', utc=True)
    
    outDict['holidaysOil']['value'] = pd.to_datetime(outDict['holidaysOil']['value'], unit='D', utc=True)
    
    outDict['usSwapIR']['date'] = pd.to_datetime(outDict['usSwapIR']['date'], unit='D', utc=True)
    
    # still need to fix usSwapCurves and Par
        
    return(outDict)

_data = _get_RT_data()

cancrudeassays = _data['cancrudeassays']
cancrudeprices = _data['cancrudeprices']
df_fut = _data['df_fut']
dflong = _data['dflong']
dfwide = _data['dfwide']
expiry_table = _data['expiry_table']
holidaysOil = _data['holidaysOil']
tickers_eia = _data['tickers_eia']

ng_storage = _data['ng_storage']
tickers_eia = _data['tickers_eia']
tradeCycle = _data['tradeCycle']
twoott = _data['twoott']
twtrump = _data['twtrump']
# usSwapCurves = convert_usSwapCurves_2p(_data['usSwapCurves'])
usSwapCurves = _data['usSwapCurves']
usSwapCurvesPar = _data['usSwapCurvesPar']
usSwapIR = _data['usSwapIR']
usSwapIRdef = _data['usSwapIRdef']
    
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
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
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
    >>> import risktools as rt
    >>> rt.simGBM(S0=5, drift=0, sigma=0.2, T=2, dt=0.25)
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
    >>> import risktools as rt
    >>> rt.simOU(S0=5,mu=5,theta=.5,sigma=0.2,T=1,dt=1/12)
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
    >>> import risktools as rt
    >>> rt.simOUJ(S0=5,mu=5,theta=.5,sigma=0.2,jump_prob=0.05,jump_avesize = 3,jump_stdv = 0.05,T=1,dt=1/12)
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
    >>> import risktools as rt
    >>> spread = rt.simOU(mu=5,theta=.5,sigma=0.2,T=5,dt=1/250)
    >>> rt.fitOU(spread)
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
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = rt.npv(init.cost=-375,C=50,cf.freq=.5,F=250,T=2,disc_factors=ir,BreakEven=True,BE_yield=.0399)
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
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = rt.npv_at_risk(init.cost=-375,C_cost=5,cf.freq=.5,F=250,T=1,disc_factors=ir,simC=[50,50,50], X=5)
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
    >>> import risktools as rt
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> spy = data.DataReader("SPY",  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> spy = spy.diff()
    >>> rt.trade_stats(df=spy['Adj Close'],Rf=0)
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

def bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "price") :
    """
    Compute bond price, cash flow table and duration
    
    Parameters
    ----------
    ytm{float}: Yield to Maturity
    C{float}: Coupon rate per annum
    T2M{float}: Time to maturity in years
    m{float}: Periods per year for coupon payments e.g semi-annual = 2.
    output{string}: "price", "df" or "duration"
    
    Returns
    -------
    Price scalar, cash flows data frame and/or duration scalar
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "price")
    >>> rt.bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "df")
    >>> rt.bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "duration")
    """
    
    out = rtl.bond(ytm, C, T2M, m, output)
    
    if (output == 'price') | (output == 'duration'):
        return out[0]
    elif output == 'df':
        return r2p(out)
    else:
        return "Error: output variable incorrect"
    
def garch(df, out=True):
    """
    Computes annualised Garch(1,1) volatilities using fGarch package.

    Parameters
    ----------
    df{DataFrame}: Wide dataframe with date column and single series (univariate). 
    out{str}: "plotly" or "matplotlib" to return respective chart types. "data" to return data or "fit" for garch fit output
    
    Returns
    -------
    out = 'data returns a pandas df
    
    Examples
    --------
    
    >>> import risktools as rt
    >>> df = rt.dflong[rt.dflong['series'] == 'CL01']
    >>> df = rt.returns(df=df,retType="rel",period_return=1,spread=True)
    >>> df = rt.rolladjust(df=df,commodityname=["cmewti"],rolltype=["Last.Trade"])
    >>> rt.garch(df,out="data")
    >>> rt.garch(df,out="fit"))
    >>> rt.garch(df,out="chart")    
    """
    
    if isinstance(df.index, pd.DatetimeIndex):
        # reset index if the df index is a datetime object
        df = df.reset_index()
        
    # R code to convert R xts format to tibble, then pandas df for out="data" or summarize fGarch obj if out = "fit"
    rFunc = """    
    garchWrap <- function(df, out) {
        library(xts)
        library(timetk)
        
        x <- RTL::garch(df, out)
        if(out=="data"){
            x <- timetk::tk_tbl(x)
            x <- rename(x, date = index)
            x <- mutate(x, date = as_date(date))                        
            }
                
        if(out=="fit"){x <- summary(x)}
                
        return(x)
    }    
    """    
    # init R code
    fnc = STAP(rFunc, "garchWrap")
    
    if (out == 'fit'):
        return r2p(fnc.garchWrap(p2r(df), out))
    elif (out == 'data'):
        ret = r2p(fnc.garchWrap(p2r(df), out))
        ret.date = pd.to_datetime(ret.date, unit='D', utc=True)
        ret = ret.set_index('date')
        return ret
    else:
        p = r2p(fnc.garchWrap(p2r(df), out))
        return image_png(p)

def returns(df,retType="abs",period_return=1,spread=False):
    """
    Computes periodic returns from a dataframe ordered by date
    
    Parameters
    ----------
    df{DataFrame}: Long dataframe with column names = ["date","value","series"]
    retType{str}: "abs" for absolute, "rel" for relative, or "log" for log returns.
    period_return{int}: Number of rows over which to compute returns.
    spread{bool}: TRUE if you want to spread into a long dataframe.
    
    Returns
    -------
    A dataframe object of returns.
    
    Examples
    --------
    >>> import risktools as rt
    >>> returns(df = rt.dflong, retType = "rel", period_return = 1, spread = True)
    >>> returns(df = rt.dflong, retType = "rel", period_return = 1, spread = False)
    """
    if isinstance(df.index, pd.DatetimeIndex):
        # reset index if the df index is a datetime object
        df = df.reset_index()
    
    ret = rtl.returns(p2r(df),retType,period_return,spread)
    ret = r2p(ret)
    ret.date = pd.to_datetime(ret.date, unit='D', utc=True)
    ret = ret.set_index('date')
    
    return r2p(ret)

def rolladjust(df,commodityname=["cmewti"],rolltype=["Last.Trade"], *args):
    """
    Returns a df adjusted for contract roll. The methodology used to adjust returns is to remove the daily returns on the day after expiry and for prices to adjust historical rolling front month contracts by the size of the roll at each expiry. This is conducive to quantitative trading strategies as it reflects the PL of a financial trader. 
    
    Parameters
    ----------
    df {DataFrame}: pandas df object of prices or returns.
    commodityname {str}: Name of commodity in expiry_table. See example below for values.
    rolltype {str}: Type of contract roll: "Last.Trade" or "First.Notice".
    *args: Other parms to pass to R function
    
    Returns
    -------
    Roll-adjusted pandas dataframe object of returns with datetime index
    
    Examples 
    --------
    >>> import risktools as rt
    >>> rt.expiry_table.cmdty.unique() # for list of commodity names
    >>> ret = rt.returns(df=rt.dflong,retType="abs",period_return=1,spread=True).iloc[:,0:1] 
    >>> rt.rolladjust(df=ret,commodityname=["cmewti"],rolltype=["Last.Trade"])
    """
    
    if isinstance(commodityname, list) == False:
        commodityname = [commodityname]
    if isinstance(rolltype,list) == False:
        rolltype = [rolltype]
        
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    ret = r2p(rtl.rolladjust(p2r(df), StrVector(commodityname), StrVector(rolltype), *args))
    ret.date = pd.to_datetime(ret.date, unit='D', utc=True)
    ret = ret.set_index('date')
    
    return ret

def prompt_beta(df, period='all',betatype='all',output = 'chart'):
    """
    Returns betas of multiple xts prices (by using relative returns).
         
    Parameters
    ----------
    x Wide dataframe with date column and multiple series columns (multivariate).
    period {str}: "all" or numeric period of time in last n periods.
    betatype {str}: "all" "bull" "bear".
    output {str} "betas", "chart","stats"
    
    Returns
    -------
    ggplot chart, df of betas or stats
        
    Examples
    --------
    import risktools as rt
    x = rt.dflong[rt.dflong.series.str.contains('CL')].copy()
    x = rt.returns(df=x,retType="abs",period_return=1,spread=True)
    x = rt.rolladjust(df=x,commodityname=["cmewti"],rolltype=["Last.Trade"])
    rt.prompt_beta(df=x,period="all",betatype="all",output="chart")
    rt.prompt_beta(df=x,period="all",betatype="all",output="betas")
    rt.prompt_beta(df=x,period="all",betatype="all",output="stats")
    """
    
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    # x = rtl.promptBeta(p2r(df), period=period, betatype=betatype, output=output)
    x = r2p(rtl.promptBeta(p2r(df), period=period, betatype=betatype, output=output))
    
    # Not sure right now...
    # R code to convert R dataframes to xts format
    rFunc = """    
    toStr <- function(x) {                
        return(summary(x))
    }
    """
    # init R code
    fnc = STAP(rFunc, "toStr")
    
    if output == 'betas':
        return x
    elif output == 'chart':                
        return image_png(x)
    
    elif output == 'stats':
        bf = dict()
        out = dict()
        
        betaformula = x[0]
                
        for i, c in enumerate(betaformula.names):
            
            if isinstance(betaformula[i], FloatVector):
                bf[c] = np.array(betaformula[i])
            else:
                bf[c] = betaformula[i]
        
        # bf = fnc.toStr(x[0])
        
        out['betaformula'] = bf
        out['betaformulaObject'] = x[1]
        
        return out
    
def chart_perf_summary(df, geometric=True, main="Cumulative Returns and Drawdowns", linesize=1.25):
    """
    Multi Asset Display of Cumulative Performance and Drawdowns

    Parameters
    ----------
    df{DataFrame}: Wide dataframe univariate or multivariate of percentage returns
    geometric{bool}: Use geometric returns True or False
    main{str}: Chart title    
    linesize{float}: Size of lines in chart and legend

    Returns
    -------
    Cumulative performance and drawdown charts as png binary

    Examples
    --------
    import risktools as rt
    df = rt.dflong[rt.dflong.series.isin(["CL01","CL12","CL36"])]
    df = rt.returns(df,retType="rel",period_return=1,spread=True)
    df = rt.rolladjust(df,commodityname=["cmewti"],rolltype=["Last.Trade"])
    rt.chart_perf_summary(df, geometric=True, main="Cumulative Returns and Drawdowns",linesize=1.25)
    """
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
        
    x = r2p(rtl.chart_PerfSummary(p2r(df), geometric, main, linesize))
    return image_png(x)

def swapIRS(trade_date = pd.Timestamp.now(),
            eff_date = pd.Timestamp.now() + pd.DateOffset(days=2),
            mat_date = pd.Timestamp.now() + pd.DateOffset(days=2) + pd.DateOffset(years=2),
            notional = 1000000,
            pay_rec = "Rec",
            fixed_rate=0.05,
            float_curve = usSwapCurves,
            reset_freq = 3,
            disc_curve = usSwapCurves,
            convention = ["act",360],
            bus_calendar="NY",
            output = "price"):
    """
    Commodity swap pricing from exchange settlement
    
    Parameters
    ----------
    trade_date {Timestamp|str}: Defaults to today().
    eff_date {Timestamp|str}: Defaults to today() + 2 days.
    mat_date {Timestamp|str}: Defaults to today() + 2 years.
    notional {long int}: Numeric value of notional. Defaults to 1,000,000.
    pay_rec {str}: "Pay" or "Rec" fixed.
    fixed_rate {float}: fixed interest rate. Defaults to 0.05.
    float_curve {R DicountCurve Obj}: List of interest rate curves. Defaults to data("usSwapCurves").
    reset_freq {int}: 1 = "monthly", 3 = quarterly, 6 = Semi annual 12 = yearly.
    disc_curve {R DicountCurve Obj}: List of interest rate curves. Defaults to data("usSwapCurves").
    convention {list}: Vector of convention e.g. c("act",360) c(30,360),...
    bus_calendar {str}: Banking day calendar. Not implemented.
    output {str}: "price" for swap price or "all" for price, cash flow data frame, duration.
    
    Returns
    -------
    Dictionary with swap price, cash flow data frame and duration.
    
    Examples
    --------
    import risktools as rt
    usSwapCurves = rt.usSwapCurves
    rt.swapIRS(trade_date = "2020-01-04", eff_date = "2020-01-06",mat_date = "2022-01-06", notional = 1000000,pay_rec = "Rec", fixed_rate=0.05, float_curve = usSwapCurves, reset_freq=3,disc_curve = usSwapCurves, convention = ["act",360],bus_calendar = "NY", output = "all")
    """
    
    # convert python dates to R
    if isinstance(trade_date,str) == True:
        tdt = base.as_Date(trade_date)
    else:
        tdt = base.as_Date(trade_date.strftime('%Y-%m-%d'))
        
    if isinstance(eff_date,str) == True:
        edt = base.as_Date(eff_date)
    else:
        edt = base.as_Date(eff_date.strftime('%Y-%m-%d'))
        
    if isinstance(mat_date,str) == True:
        mdt = base.as_Date(mat_date)
    else:
        mdt = base.as_Date(mat_date.strftime('%Y-%m-%d'))
        
            
    x = rtl.swapIRS(
        trade_date = tdt,
        eff_date = edt,
        mat_date = mdt,
        notional = notional,
        PayRec = pay_rec,
        fixed_rate = fixed_rate,
        float_curve = float_curve,
        reset_freq = reset_freq,                
        disc_curve = disc_curve,
        convention = convention,
        bus_calendar = bus_calendar,                
        output = output)
    
    if output == "price":
        out = x[0]
    else:
        out = dict()
        out['price'] = x[0][0]
        out['cashflow'] = r2p(x[1])
        out['duration'] = x[2][0]
        
        out['cashflow']['dates'] = pd.to_datetime(out['cashflow']['dates'], unit='D', utc=True)
        
    return out

##################
# r_inline_plot needed to display outputs from graphics library in R
# for chart_fwd_curves
##################

@contextmanager
def r_inline_plot(width=600, height=600, dpi=100):

    with grdevices.render_to_bytesio(grdevices.png, 
                                     width=width,
                                     height=height, 
                                     res=dpi) as b:

        yield

    data = b.getvalue()
    display(Image(data=data, format='png', embed=True))

def chart_fwd_curves(df, cmdty = 'cmewti', weekly=False, width = 1024, height = 896, dpi = 150, **kwargs):
    """    
    Returns a plot of forward curves through time
    
    Parameters
    ----------
    df{DataFrame} -- Wide dataframe with date column and multiple series columns (multivariate)
    cmdty{str} -- Futures contract code in expiry_table object: unique(expiry_table$cmdty)
    weekly{bool} -- True if you want weekly forward curves
    width {int} -- width in pixels (needed to display in IPython)
    height {int} -- height in pixels (needed to display in IPython)
    dpi {int} -- dpi in pixels (needed to display in IPython)
    *kwargs -- other NAMED graphical parameters to pass to R graphics package
    
    Returns
    -------
    plot of forward curves through time
    
    Examples
    --------
    import risktools as rt
    rt.chart_fwd_curves(df=rt.dfwide,cmdty="cmewti",weekly=True, main="WTI Forward Curves",ylab="$ per bbl",xlab="",cex=2)
    """
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    with r_inline_plot(width=width, height=height, dpi=dpi):
        r2p(rtl.chart_fwd_curves(p2r(df), cmdty, weekly, **kwargs))
    
def dist_desc_plot(df, width = 1024, height = 896, dpi = 150):
    """
    Provides a summary of returns distribution
    
    Parameters
    ----------
    df {DataFrame} -- Wide dataframe with date column/index and single series (univariate)
    width {int} -- width in pixels (needed to display in IPython)
    height {int} -- height in pixels (needed to display in IPython)
    dpi {int} -- dpi in pixels (needed to display in IPython)
    
    Returns
    -------
    Multiple plots describing the distribution
    
    Examples
    --------
    import risktools as rt
    df = rt.dflong[rt.dflong.series == "CL01"]
    df = rt.returns(df,retType="rel",period_return=1,spread=True)
    df = rt.rolladjust(df,commodityname=["cmewti"],rolltype=["Last.Trade"])
    rt.dist_desc_plot(df)
    """
    
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    with r_inline_plot(width=width, height=height, dpi=dpi):
        rtl.distdescplot(p2r(df))

# Not ready yet, eiaStocks not in main CRAN pkg 
# def chart_pairs(df, title = 'TIme Series Pairs Plot'):
#     """
#     Pairwise scatter chart for timeseries
    
#     Parameters
#     ----------
    
#     df {DataFrame} -- Wide data frame
#     title {str} -- Chart titles
    
#     Returns
#     -------
    
#     plotly figure object
    
#     Examples

#     """



# Not ready yet, eiaStocks not in main CRAN pkg    
# def chart_zscore(df, title = "NG Storage Z Score", per = "yearweek", output = "zscore", chart = "seasons"):
#     """
#     Supports analytics and display of seasonal data. Z-Score is
#     computed on residuals conditional on their seasonal period.
#     Beware that most seasonal charts in industry e.g. (NG Storage)
#     is not detrended so results once you apply an STL decompostion
#     will vary from the unajusted seasonal plot.

#     Parameters
#     ----------
    
#     df {DataFrame} -- Long data frame with columns series, date and value
#     title {str} -- Default is a blank space returning the unique value in df$series.
#     per {str} -- Frequency of seasonality "yearweek" (DEFAULT). "yearmonth", "yearquarter"
#     output {str} -- "stl" for STL decomposition chart, "stats" for STL statistical test results. "zscore" for residuals Z-score, "seasonal" for standard seasonal chart.
#     chart {str} -- "seasons" for feasts::gg_season() (DEFAULT). "series" for feasts::gg_subseries()

#     Returns
#     -------
    
#     Time series of STL decomposition residuals Z-Scores, or standard seasonal chart with feast package.

#     Examples
#     --------
    
#     import risktools as rt
#     df = rt.eiaStocks[rt.eiaStocks.series == "NGLower48"]
#     rt.chart_zscore(df = df, title = "NG Storage Z Score", per = "yearweek", output = "stl", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score", per = "yearweek", output = "stats", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score" ,per = "yearweek", output = "zscore", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score" ,per = "yearweek", output = "seasonal", chart = "seasons")
#     """

#     if isinstance(df.index,pd.DatetimeIndex):
#         df = df.reset_index()

#     x = rtl.chart_zscore(p2r(df), title, per, output, chart)
    
#     if output == "stl":
#         return image_png(x)

# def chart_pairs(df, title):
#     """
#     import risktools as rt
#     df = rt.dfwide.reset_index()[['date','CL01','NG01','HO01','RB01']]
#     """
#     x = rtl.chart_pairs(p2r(df), title)

