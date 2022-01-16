# Performance Analytics Functions

import pandas as _pd
import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

# from math import sqrt as _sqrt


def return_cumulative(r, geometric=True):
    """
    Based on the function Return.annualize from the R package PerformanceAnalytics
    by Peter Carl and Brian G. Peterson

    Calculate a compounded (geometric) cumulative return. Based om R's PerformanceAnalytics

    This is a useful function for calculating cumulative return over a period of
    time, say a calendar year.  Can produce simple or geometric return.

    product of all the individual period returns

    .. math:: (1+r_{1})(1+r_{2})(1+r_{3})\ldots(1+r_{n})-1=prod(1+R)-1

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
        r = r.add(1).prod() - 1
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
    
    .. math:: prod(1+R_{a})^{\\frac{scale}{n}}-1=\sqrt[n]{prod(1+R_{a})^{scale}}-1

    where scale is the number of periods in a year, and n is the total number of
    periods for which you have observations.

    For simple returns (geometric=FALSE), the formula is:

    .. math:: \overline{R_{a}} \cdot scale

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
    Bacon, Carl. *Practical Portfolio Performance Measurement and Attribution*. Wiley. 2004. p. 6

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
    r, scale = _check_ts(r, scale, name="r")

    r = r.dropna()
    n = r.shape[0]

    if geometric:
        res = (r.add(1).cumprod() ** (scale / n) - 1).iloc[-1]
    else:
        res = r.mean() * scale
    return res


def return_excess(R, Rf=0):
    """
    Calculates the returns of an asset in excess of the given risk free rate

    Calculates the returns of an asset in excess of the given "risk free rate"
    for the period.

    Ideally, your risk free rate will be for each period you have returns
    observations, but a single average return for the period will work too.

    Mean of the period return minus the period risk free rate

    .. math:: \overline{(R_{a}-R_{f})}

    OR

    mean of the period returns minus a single numeric risk free rate

    .. math:: \overline{R_{a}}-R_{f}

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
    if (~isinstance(x, _pd.DataFrame) & ~isinstance(x, _pd.Series)) == True:
        raise ValueError("x must be a pandas Series or DataFrame")

    if isinstance(x.index, _pd.DatetimeIndex):
        if scale is None:
            if (x.index.freq == "D") | (x.index.freq == "B"):
                scale = 252
            elif x.index.freq == "W":
                scale = 52
            elif (x.index.freq == "M") | (x.index.freq == "MS"):
                scale = 12
            elif (x.index.freq == "Q") | (x.index.freq == "QS"):
                scale = 4
            elif (x.index.freq == "Y") | (x.index.freq == "YS"):
                scale = 1
            else:
                raise ValueError(
                    "parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'"
                )
    else:
        raise ValueError(
            "parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'"
        )

    x = x.dropna()
    res = x.std() * _np.sqrt(scale)

    return res


def omega_sharpe_ratio(R, MAR, *args):
    """
    Omega-Sharpe ratio of the return distribution

    The Omega-Sharpe ratio is a conversion of the omega ratio to a ranking statistic
    in familiar form to the Sharpe ratio.

    To calculate the Omega-Sharpe ration we subtract the target (or Minimum
    Acceptable Returns (MAR)) return from the portfolio return and we divide
    it by the opposite of the Downside Deviation.

    .. math:: OmegaSharpeRatio(R,MAR) = \\frac{R_p - R_t}{ \sum^n_{t=1} \\frac{max(R_t - R_i, 0)}{n} }

    where :math:`n` is the number of observations of the entire series

    Parameters
    ----------
    R : {Series, DataFrame}
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    MAR : {float, array_like[float]}
        Minimum Acceptable Return, in the same periodicity as your
        returns
    *args : tuple
        any other passthru parameters

    Returns
    -------
    float

    Examples
    --------
    >>> mar = 0.005
    >>> print(omega_sharpe_ratio(portfolio_bacon, MAR))
    """
    if isinstance(R, (_pd.Series, _pd.DataFrame)):
        if isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if ~isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (_pd.Series, _pd.DataFrame)) & isinstance(
        R, (_pd.Series, _pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = _np.mean(MAR)

    result = (
        upside_risk(R, MAR, stat="potential")
        - downside_deviation(R, MAR, potential=True)
    ) / downside_deviation(R, MAR, potential=True)
    return result


def upside_risk(R, MAR=0, method="full", stat="risk"):
    """
    upside risk, variance and potential of the return distribution

    Upside Risk is the similar of semideviation taking the return above the
    Minimum Acceptable Return instead of using the mean return or zero
    .
    To calculate it, we take the subset of returns that are more than the target
    (or Minimum Acceptable Returns (MAR)) returns and take the differences of
    those to the target.  We sum the squares and divide by the total number of
    returns and return the square root.

    Parameters
    ----------
    R : {Series, DataFrame}
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    MAR : int
        Minimum Acceptable Return, in the same periodicity as your
        returns
    method : {'full','subset'}, default 'full'
        Either "full" or "subset", indicating whether to use the
        length of the full series or the length of the subset of the series below
        the MAR as the denominator, defaults to "full"
    stat : {"risk", "variance", "potential"}, default 'risk'
        one of "risk", "variance" or "potential" indicating whether
        to return the Upside risk, variance or potential. By default,
        'risk'

    Returns
    -------
        float

    Notes
    -----
    Equations:

    .. math:: 
        
        UpsideRisk(R, MAR) = \sqrt{\sum_{n}^{t=1} \\frac{max[(R_{t} - MAR), 0]^2}{n}}\f
        UpsideVariance(R, MAR) = \sum^{n}_{t=1} \\frac{max[(R_{t} - MAR), 0]^2}{n}\f
        UpsidePotential(R, MAR) = \sum^{n}_{t=1} \\frac{max[(R_{t} - MAR), 0]} {n}

    where `n` is either the number of observations of the entire series or
    the number of observations in the subset of the series falling below the
    MAR.

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
    # .. math:: UpsideRisk(R , MAR) = \sqrt{\sum^{n}_{t=1} \frac{max[(R_{t} - MAR), 0]^2}{n}}
    # .. math:: UpsideVariance(R, MAR) = \sum^{n}_{t=1} \frac{max[(R_{t} - MAR), 0]^2} {n}}
    # .. math:: UpsidePotential(R, MAR) = \sum^{n}_{t=1} \frac{max[(R_{t} - MAR), 0]} {n}}

    if isinstance(R, (_pd.Series, _pd.DataFrame)):
        if isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if ~isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (_pd.Series, _pd.DataFrame)) & isinstance(
        R, (_pd.Series, _pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = _np.mean(MAR)

    if method == "full":
        length = len(R)
    else:
        length = len(r)

    if stat == "risk":
        result = _np.sqrt(r.sub(MAR).pow(2).div(length).sum())
    elif stat == "variance":
        result = r.sub(MAR).pow(2).div(length).sum()
    else:
        result = r.sub(MAR).div(length).sum()

    return result


def downside_deviation(R, MAR=0, method="full", potential=False):
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

    if isinstance(R, (_pd.Series, _pd.DataFrame)):
        if isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if ~isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, _pd.DatetimeIndex) & isinstance(
            MAR, (_pd.Series, _pd.DataFrame)
        ):
            if isinstance(MAR.index, _pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.lt(MAR)]

    if isinstance(MAR, (_pd.Series, _pd.DataFrame)) & isinstance(
        R, (_pd.Series, _pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = _np.mean(MAR)

    if method == "full":
        length = len(R)
    else:
        length = len(r)

    if potential:
        result = r.mul(-1).add(MAR).div(length).sum()
    else:
        result = _np.sqrt(r.mul(-1).add(MAR).pow(2).div(length).sum())
        # result = r.mul(-1).add(MAR).pow(2).div(length).sum().apply(_np.sqrt)

    return result


def sharpe_ratio_annualized(R, Rf=0, scale=None, geometric=True):
    """
    calculate annualized Sharpe Ratio

    The Sharpe Ratio is a risk-adjusted measure of return that uses standard
    deviation to represent risk.

    The Sharpe ratio is simply the return per unit of risk (represented by
    variance).  The higher the Sharpe ratio, the better the combined performance
    of "risk" and return.

    This function annualizes the number based on the scale parameter.

    .. math:: \\frac{\sqrt[n]{prod(1+R_{a})^{scale}}-1}{\sqrt{scale}\cdot\sqrt{\sigma}}

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
    """
    R, scale = _check_ts(R, scale)

    xR = return_excess(R, Rf)
    res = return_annualized(xR, scale=scale, geometric=geometric)
    res /= sd_annualized(R, scale=scale)

    return res


def drawdowns(R, geometric=True):
    """
    Function to calculate drawdown levels in a timeseries

    Parameters
    ----------
    R : {Series, DataFrame}
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True
    """

    if geometric:
        res = R.add(1).cumprod()
    else:
        res = R.cumsum() + 1

    res = res / res.clip(lower=1).cummax() - 1

    return res


def find_drawdowns(R, geometric=True, *args):
    """
    Find the drawdowns and drawdown levels in a timeseries.

    find_drawdowns() will find the starting period, the ending period, and
    the amount and length of the drawdown.

    Often used with sort_drawdowns() to get the largest drawdowns.

    drawdowns() will calculate the drawdown levels as percentages, for use
    in \code{\link{chart.Drawdown}}.

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True

    Returns
    -------
    Nested dictionary with asset name(s) as the top level key(s). Nested below that as another
    dictionary are the following keys and values:

    'return'- numpy array of minimum of returns below the risk free rate of return (Rf) for each
        trough. If returns are positive, array has 0 value
    'from' - array index positiion of beginning of each trough or recovery period corresponding to each
        element of 'return'
    'trough' - array index position of peak trough period corresponding to each
        element of 'return'. Returns beginning of recovery periods
    'to' - array index positiion of end of each trough or recovery period corresponding to each
        element of 'return'
    'length' - length of each trough period corresponding to each element of 'return' as given by
        the difference in to and from index positions
    'peaktotrough' - array index distance from the peak of each trough or recovery period from the
        beginning of each trough or recovery period, corresponding to each element of 'return'
    'recovery' - array index distance from the peak of each trough or recovery period to the
        end of each trough or recovery period, corresponding to each element of 'return'

    References
    ----------
    Bacon, C. *Practical Portfolio Performance Measurement and Attribution*. Wiley. 2004. p. 88

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
    if isinstance(dd, _pd.Series):
        dd = _pd.DataFrame({"drawdown": dd})
        series_flag = True

    for lab, con in dd.iteritems():
        rs[lab] = dict()
        rs[lab]["return"] = _np.array([]).astype(float)
        rs[lab]["from"] = _np.array([]).astype(int)
        rs[lab]["to"] = _np.array([]).astype(int)
        rs[lab]["length"] = _np.array([]).astype(int)
        rs[lab]["trough"] = _np.array([]).astype(int)

        if con[0] >= 0:
            prior_sign = 1
        else:
            prior_sign = 0

        frm = 0
        to = 0
        dmin = 0
        sofar = con[0]

        for i, r in enumerate(con):  # .iteritems():
            if r < 0:
                this_sign = 0
            else:
                this_sign = 1

            if this_sign == prior_sign:
                if r < sofar:
                    sofar = r
                    dmin = i
                to = i + 1
            else:
                rs[lab]["return"] = _np.append(rs[lab]["return"], sofar)
                rs[lab]["from"] = _np.append(rs[lab]["from"], frm)
                rs[lab]["trough"] = _np.append(rs[lab]["trough"], dmin)
                rs[lab]["to"] = _np.append(rs[lab]["to"], to)

                frm = i
                sofar = r
                to = i + 1
                dmin = i
                prior_sign = this_sign

        rs[lab]["return"] = _np.append(rs[lab]["return"], sofar)
        rs[lab]["from"] = _np.append(rs[lab]["from"], frm)
        rs[lab]["trough"] = _np.append(rs[lab]["trough"], dmin)
        rs[lab]["to"] = _np.append(rs[lab]["to"], to)

        rs[lab]["length"] = rs[lab]["to"] - rs[lab]["from"] + 1
        rs[lab]["peaktotrough"] = rs[lab]["trough"] - rs[lab]["from"] + 1
        rs[lab]["recovery"] = rs[lab]["to"] - rs[lab]["trough"]

        # if original parameter was a series, remove top layer of
        # results dictionary
        if series_flag == True:
            rs = rs["drawdown"]

    return rs


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
        subset = _np.repeat(True, len(x))
    else:
        subset = subset.dropna().astype(bool)

    if (
        (isinstance(x, (_np.ndarray, _pd.Series)) == False)
        & (isinstance(y, (_np.ndarray, _pd.Series)) == False)
        & (isinstance(subset, (_np.ndarray, _pd.Series)) == False)
    ):
        raise ValueError(
            "all arguements of _beta must be pandas Series or numpy arrays"
        )

    # convert to arrays
    x = _np.array(x)
    y = _np.array(y)
    subset = _np.array(subset)

    # subset
    x = x[subset]
    y = y[subset]

    model = _LinearRegression()
    model.fit(x.reshape((-1, 1)), y)
    beta = model.coef_

    return beta[0]


def CAPM_beta(Ra, Rb, Rf=0, kind="all"):
    """
    calculate single factor model (CAPM) beta

    The single factor model or CAPM Beta is the beta of an asset to the variance
    and covariance of an initial portfolio.  Used to determine diversification potential.

    This function uses a linear intercept model to achieve the same results as
    the symbolic model used by \code{\link{BetaCoVariance}}

    .. math:: \\beta_{a,b}=\\frac{CoV_{a,b}}{\sigma_{b}}=\\frac{\sum((R_{a}-\\bar{R_{a}})(R_{b}-\\bar{R_{b}}))}{\sum(R_{b}-\\bar{R_{b}})^{2}}

    Ruppert(2004) reports that this equation will give the estimated slope of
    the linear regression of :math:`R_{a}` on :math:`R_{b}` and that this
    slope can be used to determine the risk premium or excess expected return
    (see Eq. 7.9 and 7.10, p. 230-231).

    kind='bull' and kind='bear' apply the same notion of best fit to positive and
    negative market returns, separately. kind='bull' is a
    regression for only positive market returns, which can be used to understand
    the behavior of the asset or portfolio in positive or 'bull' markets.
    Alternatively, kind='bear' provides the calculation on negative
    market returns.

    The function `timing_ratio` may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio, which is calculated as
    
    :math:`TimingRatio =\\frac{\\beta^{+}}{\\beta^{-}}`
    
    is best when greater than one in a rising market and less than one in a
    falling market.

    While the classical CAPM has been almost completely discredited by the
    literature, it is an example of a simple single factor model,
    comparing an asset to any arbitrary benchmark.

    Parameters
    ----------
    Ra : {array_like | DataFrame}
        Array-like or DataFrame with datetime index of asset returns to be tested vs benchmark
    Rb : array_like
        Benchmark returns to use to test Ra
    Rf : {array_like | float}
        risk free rate, in same period as your returns, or as a single
        digit average
    kind : {'all','bear','bull'}, default 'all'
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
    Sharpe, W.F. Capital Asset Prices: A theory of market equilibrium under conditions of risk. *Journal of finance*, vol 19, 1964, 425-442.\f
    Ruppert, David. *Statistics and Finance, an Introduction*. Springer. 2004.\f
    Bacon, Carl. *Practical portfolio performance measurement and attribution*. Wiley. 2004.

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

    if kind == "bear":
        subset = xRb.lt(0).mask(
            xRb.isna(), _np.nan
        )  # need to include the mask, otherwise lt/gt returns False for NaN
    elif kind == "bull":
        subset = xRb.gt(0).mask(
            xRb.isna(), _np.nan
        )  # need to include the mask, otherwise lt/gt returns False for NaN
    else:
        subset = None

    if isinstance(xRa, _pd.DataFrame):
        # applies function _beta to each columns of df
        rs = xRa.apply(lambda x: _beta(x, xRb, subset), axis=0)
    else:
        rs = _beta(xRa, xRb, subset)

    return rs


def timing_ratio(Ra, Rb, Rf=0):
    """
    The function `timing_ratio` may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio is best when greater than one in a 
    rising market and less than one in a falling market.

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

    Notes
    -----

    .. math:: TimingRatio = \\frac{\\beta^{+}}{\\beta^{-}}

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

    beta_bull = CAPM_beta(Ra, Rb, Rf=Rf, kind="bull")
    beta_bear = CAPM_beta(Ra, Rb, Rf=Rf, kind="bear")

    result = beta_bull / beta_bear

    return result


def _check_ts(R, scale, name="R"):
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
    if (~isinstance(R, _pd.DataFrame) & ~isinstance(R, _pd.Series)) == True:
        raise ValueError(f"{name} must be a pandas Series or DataFrame")

    if isinstance(R.index, _pd.DatetimeIndex):
        if scale is None:
            if (R.index.freq == "D") | (R.index.freq == "B"):
                scale = 252
            elif R.index.freq == "W":
                scale = 52
            elif (R.index.freq == "M") | (R.index.freq == "MS"):
                scale = 12
            elif (R.index.freq == "Q") | (R.index.freq == "QS"):
                scale = 4
            elif (R.index.freq == "Y") | (R.index.freq == "YS"):
                scale = 1
            else:
                raise ValueError(
                    f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'"
                )
    else:
        raise ValueError(
            f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'"
        )

    return R, scale
