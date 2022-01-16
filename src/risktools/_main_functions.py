import warnings
import pandas as _pd
import numpy as _np
import quandl as _quandl
from . import data
import arch as _arch
from scipy.optimize import least_squares as _least_squares
from scipy import interpolate as _interpolate
import plotly.express as _px
import plotly.graph_objects as _go
import matplotlib.pyplot as _plt
from ._morningstar import *
from statsmodels.tsa.seasonal import STL as _STL
from pandas.plotting import (
    register_matplotlib_converters as _register_matplotlib_converters,
)
import seaborn as _sns

from ._pa import *


def ir_df_us(quandl_key=None, ir_sens=0.01, date=None):
    """
    Extracts US Tresury Zero Rates using Quandl

    Parameters
    ----------
    quandl_key : str
        Your Quandl key "yourkey" as a string, by default None. Optional as Quandl allows up to 50
        calls per day for free
    ir_sens : float
        Creates plus and minus IR sensitivity scenarios with specified shock value.
    date : string
        If date is not none, function will return ir curve for that date. By default None.

    Returns
    -------
    A pandas data frame of zero rates

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us()
    """

    if date is None:
        edt = _pd.Timestamp.now().floor("D")
    else:
        edt = _pd.to_datetime(date)
    # Last 30 days
    sdt = edt - _pd.DateOffset(days=30)

    if quandl_key is not None:
        _quandl.ApiConfig.api_key = quandl_key

    zero_1yr_plus = _quandl.get("FED/SVENY", start_date=sdt, end_date=edt)

    if zero_1yr_plus.index.max() < edt:
        warnings.warn(
            f"data only available until {zero_1yr_plus.index.max()}. Pulling data for {zero_1yr_plus.index.max()}"
        )
        edt = zero_1yr_plus.index.max()
        sdt = edt - _pd.DateOffset(days=30)

    fedsfund = _quandl.get("FED/RIFSPFF_N_D", start_date=sdt, end_date=edt).dropna()
    # print(fedsfund)
    fedsfund["FedsFunds0"] = _np.log((1 + fedsfund.Value / 360) ** 365)
    fedsfund.drop("Value", axis=1, inplace=True)

    zero_tb = _quandl.get(
        ["FED/RIFLGFCM01_N_B", "FED/RIFLGFCM03_N_B", "FED/RIFLGFCM06_N_B"],
        start_date=sdt,
        end_date=edt,
    ).dropna()
    zero_tb.columns = zero_tb.columns.str.replace(" - Value", "", regex=False)

    # get most recent full curve (some more recent days will have NA columns)
    x = fedsfund.join(zero_tb).join(zero_1yr_plus).dropna().iloc[-1, :].reset_index()
    x.columns = ["maturity", "yield"]
    x["index"] = x["maturity"]
    x["yield"] /= 100
    x["maturity"] = x.maturity.str.extract("(\d+)").astype("int")

    # change maturity numbers to year fraction for first four rows
    x.iloc[1:4, x.columns.get_loc("maturity")] /= 12
    x.iloc[0, x.columns.get_loc("maturity")] = 1 / 365
    # x.maturity[1:4] /= 12
    # x.maturity[0] = 1/365

    # add new row for today, same yield as tomorrow
    x = _pd.concat(
        [_pd.DataFrame({"maturity": [0], "yield": [x["yield"][0]]}), x],
        ignore_index=True,
    )

    x["discountfactor"] = _np.exp(-x["yield"] * x.maturity)
    x["discountfactor_plus"] = _np.exp(-(x["yield"] + ir_sens) * x.maturity)
    x["discountfactor_minus"] = _np.exp(-(x["yield"] - ir_sens) * x.maturity)

    x.loc[0, "index"] = "0"
    x = x.set_index("index")

    x = x[
        [
            "yield",
            "maturity",
            "discountfactor",
            "discountfactor_plus",
            "discountfactor_minus",
        ]
    ]

    return x


def bond(ytm=0.05, c=0.05, T=1, m=2, output="price"):
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
    assert output in [
        "df",
        "price",
        "duration",
    ], "output not a member of ['df','price','duration']"

    # init df
    df = _pd.DataFrame(
        {"t_years": _np.arange(1 / m, T + 1 / m, 1 / m), "cf": [c * 100 / m] * (T * m)}
    )
    df["t_periods"] = df.t_years * m
    df.loc[df.t_years == T, "cf"] = c * 100 / m + 100
    df["disc_factor"] = 1 / ((1 + ytm / m) ** df["t_periods"])
    df["pv"] = df.cf * df.disc_factor
    price = df.pv.sum()
    df["duration"] = df.pv * df.t_years / price

    if output == "price":
        ret = price
    elif output == "df":
        ret = df
    elif output == "duration":
        ret = df.duration.sum()
    else:
        ret = None

    return ret


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
    >>> from pandas_datareader import data
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", '2000-01-01', '2012-01-01')
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> rt.trade_stats(df[('Adj Close','SPY')])
    """

    # r = R.dropna() # don't use, messes up freq check
    r = R.copy()

    rs = dict()

    # convert series into dataframe for flexibility
    series_flag = False
    if isinstance(r, _pd.Series):
        r = _pd.DataFrame({"trade_stats": r})
        series_flag = True

    for lab, con in r.iteritems():
        rs[lab] = dict()
        y = find_drawdowns(con)
        rs[lab]["cum_ret"] = return_cumulative(con, geometric=True)
        rs[lab]["ret_ann"] = return_annualized(con, scale=252)
        rs[lab]["sd_ann"] = sd_annualized(con, scale=252)
        rs[lab]["omega"] = omega_sharpe_ratio(con, MAR=Rf)  # * _np.sqrt(252)
        rs[lab]["sharpe"] = sharpe_ratio_annualized(con, Rf=Rf)

        # need to dropna to calc perc_win properly
        con_clean = con.dropna()
        rs[lab]["perc_win"] = (
            con_clean[con_clean > 0].shape[0] / con_clean[con_clean != 0].shape[0]
        )
        rs[lab]["perc_in_mkt"] = con_clean[con_clean != 0].shape[0] / con_clean.shape[0]

        rs[lab]["dd_length"] = max(y["length"])
        rs[lab]["dd_max"] = min(y["return"])

    if series_flag == True:
        rs = rs["trade_stats"]

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

    if isinstance(df, _pd.Series):
        df = _pd.DataFrame({df.name: df})
    elif isinstance(df, _pd.DataFrame) == False:
        raise ValueError("df is not a pandas Series or DataFrame")

    index_flag = False
    if len(df.index.names) == 1:
        # convert single index into multi-index by adding dummy index in position 0 to allow for groupby
        df["dummy"] = "dummy"
        df = df.set_index("dummy", append=True)
        df = df.swaplevel()
        index_flag = True

    if (spread == True) & (len(df.index.names) == 1):
        raise ValueError(
            "You have set spread to True but only passed a single index, not a multi-index. There is nothing to spread"
        )

    # df = df.dropna()

    # calc return type
    if ret_type == "abs":
        df = df.groupby(level=0).apply(lambda x: x.diff())
    elif ret_type == "rel":
        df = df.groupby(level=0).apply(lambda x: x / x.shift(period_return) - 1)
    elif ret_type == "log":
        if df[df < 0].count().sum() > 0:
            warnings.warn(
                "Negative values passed to log returns. You will likely get NaN values using log returns",
                RuntimeWarning,
            )
        df = df.groupby(level=0).apply(lambda x: _np.log(x / x.shift(period_return)))
    else:
        raise ValueError("ret_type is not valid")

    if spread:
        # return spread df, don't dropna or it will remove entire rows
        df = df.unstack(level=0).droplevel(level=0, axis=1)
        return df
    else:
        if index_flag == True:
            # drop dummy index to return single index
            df = df.droplevel(0)
        if len(df.columns) == 1:
            return df.dropna().iloc[:, 0]
        else:
            return df.dropna()


def roll_adjust(
    df, commodity_name="cmewti", roll_type="Last_Trade", roll_sch=None, *args
):
    """
    Returns a pandas series adjusted for contract roll. The methodology used to adjust returns is to remove the daily returns on
    the day after expiry and for prices to adjust historical rolling front month contracts by the size of the roll at
    each expiry. This is conducive to quantitative trading strategies as it reflects the PL of a financial trader.

    Note that this will apply a single expiry schedule to all assets passed to this function. To apply different roll schedules
    by asset type, perform a separate function call.

    Parameters
    ----------
    df : Series | DataFrame
        pandas series or dataframe with a with a datetime index and which columns are asset prices or with a dataframe with a multi-index in the shape ('asset name','date')
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

    if isinstance(df, _pd.Series):
        df = _pd.DataFrame({df.name: df})
    elif isinstance(df, _pd.DataFrame) == False:
        raise ValueError("df is not a pandas Series or DataFrame")

    if roll_sch is None:
        roll_sch = data.open_data("expiry_table")
        roll_sch = roll_sch[roll_sch.cmdty == commodity_name]
        roll_sch = roll_sch[roll_type]

    df["expiry"] = df.index.isin(roll_sch, level=-1)
    df["expiry"] = df["expiry"].shift()

    df = df[df.expiry == False].drop("expiry", axis=1)

    if len(df.columns) == 1:
        return df.dropna().iloc[:, 0]
    else:
        return df.dropna()


def garch(df, out="data", scale=None, show_fig=True, forecast_horizon=1, **kwargs):
    """
    Computes annualised Garch(1,0,1) volatilities using arch package. Note that the
    RTL package uses a sGarch model, but there is no similiar implementation in
    python.

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
    df = _check_df(df).sort_index()

    # if kwargs is None:
    #     kwargs = {"p": 1, "o": 0, "q": 1}

    df = df - df.mean()

    # find a more robust way to do this, rolladjust may break it
    freq = _pd.infer_freq(df.index[-10:])

    if scale is None:
        if freq is None:
            raise ValueError(
                "Could not infer frequency of timeseries, please provide scale parameter instead"
            )
        elif (freq == "B") or (freq == "D"):
            scale = 252
        elif freq[0] == "W":
            scale = 52
        elif (freq == "M") or (freq == "MS"):
            scale = 12
        elif (freq == "Q") or (freq == "QS"):
            scale = 4
        elif (freq == "Y") or (freq == "YS"):
            scale = 1
        else:
            raise ValueError(
                "Could not infer frequency of timeseries, please provide scale parameter instead"
            )

    # a standard GARCH(1,1) model
    garch = _arch.arch_model(df, **kwargs)
    garch_fitted = garch.fit()

    # # calc annualized volatility from variance
    yhat = _np.sqrt(
        garch_fitted.forecast(horizon=forecast_horizon, start=0, reindex=True).variance
    ) * _np.sqrt(scale)

    if out == "data":
        return yhat
    elif out == "fit":
        return garch_fitted
    elif out == "plotly":
        fig = _px.line(yhat)
        fig.show()
        return fig
    elif out == "matplotlib":
        fig, ax = _plt.subplots(1, 1)
        yhat.plot(ax=ax)
        fig.show()
        return fig, ax


def prompt_beta(df, period="all", beta_type="all", output="chart"):
    """
    Returns array/dataframe of betas for futures contract returns of a commodity
    with it's front contract (i.e. next most expirying contract). For use with futures
    contracts (i.e. NYMEX WTI: CL01, CL02, CL03 and so forth) with standardized expiry periods.

    Using the WTI example, the betas will represent the covariance of daily returns of
    CL02, CL03, CL04 with CL01. The covariance of CL01 with CL01 is also returned, but is
    always 1.

    This function uses the single factor model (CAPM) beta. See the function CAPM_beta
    for more details

    Parameters
    ----------
    df : DataFrame
        Wide dataframe with datetime index and multiple series columns for each futures contract.
        Always use continuous contracts for columns.
    period : [str | int | float]
        Timeframe to use to calculate beta. "all" to use all data available or scalar number
        n to only use the last n periods/rows of data from the last, by default 'all'. i.e.
        for WTI contracts (CL), 30 would be the last 30 days. Recommend running roll_adjust
        function prior to using prompt_beta to remove swings from contract expiry
    beta_type : {'all', 'bull', 'bear'}, default 'all'
        Beta types to return
    output : {'betas', 'chart', 'stats'}, default 'chart'
        Output type

    Returns
    -------
    chart
        A plotly figure with the beta lines charted for 'all', 'bear' and 'bull' markets
    betas
        A dataframe of betas by contract order for 'all', 'bear' and 'bull' markets
    stats
        A scipy object from a least_squares fit of the betas by market type. Model used
        to fit betas was of the form:
        .. math:: \{beta} = x0 * exp(x1*t) + x2
        where t is the contract order (1, 2, 3 etc..., lower for expirying sooner)

    chart, df of betas or stats

    Examples
    --------
    >>> import risktools as rt
    >>> dfwide = rt.data.open_data('dfwide')
    >>> col_mask = dfwide.columns[dfwide.columns.str.contains('CL')]
    >>> dfwide = dfwide[col_mask]
    >>> x = rt.returns(df=dfwide, ret_type="abs", period_return=1)
    >>> x = rt.roll_adjust(df=x, commodity_name="cmewti", roll_type="Last_Trade")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="chart")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="betas")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="stats")
    >>> rt.prompt_beta(df=x, period=30, beta_type="all", output="plotly")
    """
    df = df.copy()
    # this assumes that the numeric component of the column name represents
    # an order to the asset contract
    term = df.columns.str.replace("[^0-9]", "", regex=True).astype(int)

    if isinstance(period, (int, float)):
        df = df.sort_index()
        df = df.iloc[-period:]

    # leave only contract # in column names
    df.columns = term
    df = df.sort_index()

    # calculate betas by market type (mkt is all types) using front contract as the benchmark
    mkt = CAPM_beta(df, df.iloc[:, 0])
    bull = CAPM_beta(df, df.iloc[:, 0], kind="bull")
    bear = CAPM_beta(df, df.iloc[:, 0], kind="bear")

    # create array for non-linear least squares exponential
    prompt = _np.arange(0, mkt.shape[0]) + 1

    # proposed model for beta as a function of prompt
    def beta_model(x, prompt):
        return x[0] * _np.exp(x[1] * prompt) + x[2]

    # cost function for residuals. Final equation that we're trying to minimize is
    # beta - x0*exp(x1*prompt) + x2
    def beta_residuals(x, beta, prompt):
        return beta - beta_model(x, prompt)

    # run least squares fit. Note that I ignore the first row of mkt and prompt arrays since
    # correlation of a var with itself should always be 1. Also, the beta of the second contract
    # will likely be a lot less then 1, and so ignoring the 1st contract will allow for a better fit
    r = _least_squares(
        beta_residuals, x0=[-1, -1, -1], args=(_np.array(mkt[1:]), prompt[1:])
    )

    # construct output df
    out = _pd.DataFrame()
    out["all"] = mkt
    out["bull"] = bull
    out["bear"] = bear

    if output == "stats":
        return r
    elif output == "betas":
        return out
    elif output == "chart":
        out = out[
            1:
        ]  # exclude beta of front contract with itself (always 1, no information)
        fig = _go.Figure()
        fig.add_trace(_go.Scatter(x=out.index, y=out["all"], mode="lines", name="all"))
        fig.add_trace(
            _go.Scatter(x=out.index, y=out["bear"], mode="lines", name="bear")
        )
        fig.add_trace(
            _go.Scatter(x=out.index, y=out["bull"], mode="lines", name="bear")
        )
        fig.update_xaxes(range=[out.index.min() - 1, out.index.max() + 1])
        fig.update_layout(
            title="Contract Betas vs Front Contract: Bear (Bull) = Beta in Down (Up) Moves",
            xaxis_title="Contract",
            yaxis_title="Beta",
            legend_title="Market Type",
        )
        return fig


def npv(
    init_cost=-375,
    C=50,
    cf_freq=0.25,
    F=250,
    T=2,
    disc_factors=None,
    break_even=False,
    be_yield=0.01,
):
    """
    Compute Net Present Value using discount factors from ir_df_us()

    Parameters
    ----------
    init_cost : float | int
        Initial investment cost
    C : float | int
        Periodic cash flow
    cf_freq : float
        Cash flow frequency in year fraction e.g. quarterly = 0.25
    F : float | int
        Final terminal value
    T : float | int
        Final maturity in years
    disc_factors : DataFrame
        Data frame of discount factors using ir_df_us() function.
    break_even : bool
        True when using a flat discount rate assumption.
    be_yield : float | int
        Set the flat IR rate when beak_even = True.

    Returns
    -------
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(ir_sens=0.01)
    >>> rt.npv(init_cost=-375, C=50, cf_freq=0.5, F=250, T=2, disc_factors=ir, break_even=True, be_yield=.0399)
    """
    disc_factors = disc_factors.copy()

    if disc_factors is None:
        raise ValueError(
            "Please input a discount factor dataframe into disc_factors (use ir_df_us to get dataframe)"
        )

    if break_even == True:
        print("test")
        disc_factors["yield"] = be_yield
        disc_factors["discountfactor"] = _np.exp(
            -disc_factors["yield"] * disc_factors.maturity
        )

    n = len(_np.arange(0, T, cf_freq)) + 1

    disc_intp = _interpolate.splrep(disc_factors.maturity, disc_factors.discountfactor)

    df = _pd.DataFrame(
        {
            "t": _np.append(
                _np.arange(0, T, cf_freq), [T]
            ),  # need to append T since _np.arange is of type [a,b)
            "cf": _np.ones(n) * C,
        }
    )

    df.loc[df.t == 0, "cf"] = init_cost
    df.loc[df.t == T, "cf"] = F

    df["df"] = _interpolate.splev(df.t, disc_intp)
    df["pv"] = df.cf * df.df

    return df


def crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call"):
    """
    European option binomial model on a stock without dividends. For academic purposes only.
    Use fOptions::CRRBinomialTreeOptions for real-life usage in R or Python equivalent.

    Parameters
    ----------
    s : float
        Stock price, by default 100
    x : float
        Strike price, by default 100
    sigma : float
        Implied volatility, by default 0.20
    Rf : float
        Risk-free rate, by default 0.1
    T : float
        Time to maturity in years, by default 1
    n :
        Number of time steps, by default 5. Internally dt = T/n.
    type : str
        "call" or "put", by default "call"

    Returns
    -------
    List of asset price tree, option value tree and option price.

    Examples
    --------
    >>> import risktools as rt
    >>> rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")
    """

    dt = T / n

    # define u, d, and risk-neutral probability
    u = _np.exp(sigma * _np.sqrt(dt))
    d = _np.exp(-sigma * _np.sqrt(dt))
    q = (_np.exp(Rf * dt) - d) / (u - d)

    # define our asset tree prices
    asset = _np.zeros([n + 1, n + 1])

    for i in range(0, n + 1):
        for j in range(0, i + 1):
            asset[i, j] = s * (u ** j) * (d ** (i - j))

    # create matrix of the same dims as asset price tree
    option = _np.zeros([n + 1, n + 1])
    # replace last row with maturity payoffs
    if type == "call":
        option[-1, :] = asset[-1, :] - x
    elif type == "put":
        option[-1, :] = x - asset[-1, :]
    else:
        raise ValueError("type is not of ('call','put')")
    option[option < 0] = 0

    # we discount recursively starting from final payoff (last row of tree)
    # starting at the second last period based on final payoffs
    for i in range(n - 1, -1, -1):  # loop for n-1 to 0 because range(a,b) is [a,b)
        for j in range(0, i + 1):
            option[i, j] = (
                (1 - q) * option[i + 1, j] + q * option[i + 1, j + 1]
            ) / _np.exp(Rf * dt)

    # indicator if model can be used sigma > rsqrt(dt)
    if sigma > _np.sqrt(dt) * Rf:
        note = "ok"
    else:
        note = "sigma < Rf*sqrt(dt) do not use"

    return {"asset": asset, "option": option, "price": option[0, 0], "note": note}


def stl_decomposition(
    df,
    output="chart",
    period=None,
    seasonal=7,
    seasonal_deg=1,
    resample_freq="M",
    **kwargs,
):
    """
    Provides a summary of returns distribution using the statsmodels.tsa.seasonal.STL class.
    Resamples all data to  prior to STL if

    Parameters
    ----------
    df : Series
        A pandas series (univariate) with a datetime index.
    output : str
        "chart" to see output as a graph, "data" for results as a list. By default "chart"
    seasonal : int, optional
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7 (default).
    seasonal_deg : int, optional
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend). By default 1
    resample_freq : str
        Resampling frequency to use in pandas resample function. Set to None to not resample
    **kwargs :
        Other parms



    season - The length of the seasonal smoother. Must be odd.

    trend - The length of the trend smoother, usually around 150% of season. Must be odd and larger than season.

    low_pass - The length of the low-pass estimation window, usually the smallest odd number larger than the periodicity of the data.


    Returns
    -------
    a chart or statsmodels.tsa.seasonal.DecomposeResult object of results

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.open_data('dflong')
    >>> df = df['CL01']
    >>> rt.stl_decomposition(df, output="chart", seasonal=13, seasonal_deg=1, resample_freq='M')
    >>> rt.stl_decomposition(df, output="data", seasonal=13, seasonal_deg=1, resample_freq='M')
    """

    _register_matplotlib_converters()
    _sns.set_style("darkgrid")

    if resample_freq is not None:
        df = df.resample(resample_freq).mean()

    stl = _STL(
        df,
        period=period,
        seasonal=seasonal,
        seasonal_deg=seasonal_deg,
        robust=True,
        **kwargs,
    )
    res = stl.fit()

    if output == "chart":
        fig = res.plot()
        fig.set_size_inches(15, 15)
        return fig  # res.plot()
    else:
        return res


def get_eia_df(tables, key):
    """
    Function for download data from the US Government EIA and return it as a pandas dataframe/series

    To get an api key, go to https://www.eia.gov/opendata/register.php
    To get table names, search via API explorer at https://www.eia.gov/opendata/qb.php

    Parameters
    ----------
    tables : List[Tuple[str]]
        EIA series to return. Can be a list or tuple of tables as well.
    key : str
        EIA key.

    Returns
    -------
    pandas dataframe or series depending on the number of tables requested

    Examples
    --------
    >>> import risktools as rt
    >>> rt.get_eia_df('PET.WDIIM_R20-Z00_2.W', key=eia_key)
    """
    import requests
    import json

    if isinstance(tables, list) == False:
        tables = [tables]

    eia = _pd.DataFrame()

    for tbl in tables:
        url = r"http://api.eia.gov/series/?api_key={}&series_id={}&out=json".format(
            key, tbl
        )
        tmp = json.loads(requests.get(url).text)

        tf = _pd.DataFrame(tmp["series"][0]["data"], columns=["date", "value"])
        tf["table_name"] = tmp["series"][0]["name"]
        tf["series_id"] = tmp["series"][0]["series_id"]
        eia = eia.append(tf)

    eia.loc[eia.date.str.len() < 7, "date"] += "01"

    eia.date = _pd.to_datetime(eia.date)
    return eia


def _check_df(df):
    # if isinstance(df.index, _pd.DatetimeIndex):
    #     # reset index if the df index is a datetime object
    #     df = df.reset_index().copy()

    return df.copy()


def infer_freq(x, multiplier=False):
    """
    Function to infer the frequency of a time series. Improvement over 
    pandas.infer_freq as it can handle missing days/holidays. Note that
    for business days it will return 'D' vs 'B'

    Parameters
    ----------
    x : DataFrame | Series
        Time series. Index MUST be a datetime index.
    multiplier : bool
        If True, returns annualization factor for financial time series:
        252 for daily/business day
        52 for weekly
        12 for monthly
        4 for quarterly
        1 for annual
    Returns
    -------
    str is multiplier = False
    """

    # searches for 3 consecutive rows and then infers freq
    x = x.copy()
    x.index = _pd.to_datetime(x.index)
    diffs = x.index[1:] - x.index[:-1]
    min_delta = diffs.min()
    mask = (diffs == min_delta)[:-1] & (diffs[:-1] == diffs[1:])
    pos = _np.where(mask)[0][0]

    freq = _pd.infer_freq(x.index[pos : pos + 3])
    if multiplier == False:
        return freq
    else:
        if freq in ["D", "B"]:
            return 252
        elif freq[0] == "W":
            return 52
        elif freq[0] == "M":
            return 12
        elif freq[0] == "Q":
            return 4
        elif freq[0] == "A":
            return 1

