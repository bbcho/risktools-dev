from . import data

import pandas as _pd
import numpy as _np
from scipy import interpolate as _interpolate
from ._morningstar import *

import pandas_datareader as _pdr
import plotly.express as _px

us_swap = data.open_data("usSwapCurves")


def custom_date_range(start, end, freq):
    if freq == "M":
        mul = 1
    elif freq == "Q":
        mul = 3
    elif (freq == "H") | (freq == "6M"):
        mul = 6
    elif freq == "Y":
        mul = 12
    else:
        raise ValueError("freq is not in ['M','Q','H' or 'Y']")

    dr = [_pd.to_datetime(start)]

    i = 1
    while dr[-1] < _pd.to_datetime(end):
        dr.append(dr[0] + _pd.DateOffset(months=mul * i))
        i += 1

    return _pd.Index(dr)


def swap_irs(
    trade_date=None,
    eff_date=None,
    mat_date=None,
    notional=1000000,
    pay_rec="rec",
    fixed_rate=0.05,
    float_curve=None,
    reset_freq="Q",
    disc_curve=None,
    days_in_year=360,
    convention="act",
    bus_calendar="NY",  # not implemented
    output="price",
):
    """
    Commodity swap pricing from exchange settlement

    Parameters
    ----------
    trade_date : Timestamp | str
        The date of the trade. If None, defaults to today(). By default None.
    eff_date : Timestamp | str
        The effective date of the swap. If None, defaults to trade_date + 2 days. By default None.
    mat_date : Timestamp | str
        The maturity date of the swap. If None, defaults to eff_date + 2 years.
    notional : long int
        Numeric value of notional. Defaults to 1,000,000.
    pay_rec : str
        "pay" for pyement (positive pv) or "rec" for receivables (negative pv).
    fixed_rate : float
        fixed interest rate. Defaults to 0.05.
    float_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. If None, defaults to rt.data.open_data("usSwapCurves")
        which is a dictionary based on the R object DiscountCurve. Column times is year fractions going forward in time.
        So today is 0, tomorrow is 1/365, a year from today is 1 etc...
    reset_freq : str
        Pandas/datetime Timestamp frequency (allowable values are 'M', 'Q', '6M', or 'Y')
    disc_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. If None, defaults to rt.data.open_data("usSwapCurves")
        which is a dictionary based on the R object DiscountCurve. Column times is year fractions going forward in time.
        So today is 0, tomorrow is 1/365, a year from today is 1 etc...
    days_in_year : int
        360 or 365
    convention : list
        List of convention e.g. ["act",360], [30,360],...
    bus_calendar : str
        Banking day calendar. Not implemented.
    output : str
        "price" for swap price or "all" for price, cash flow data frame, duration.

    Returns
    -------
    returns present value as a scalar if output=='price. Otherwise returns dictionary with swap price, cash flow data frame and duration

    Examples
    --------
    >>> import risktools as rt
    >>> usSwapCurves = rt.data.open_data('usSwapCurves')
    >>> rt.swap_irs(trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06", notional=1000000, pay_rec = "rec", fixed_rate=0.05, float_curve=usSwapCurves, reset_freq='Q', disc_curve=usSwapCurves, days_in_year=360, convention="act", bus_calendar="NY", output = "all")
    """
    if trade_date is None:
        trade_date = (_pd.Timestamp.now().floor("D"),)
    if eff_date is None:
        eff_date = trade_date + _pd.DateOffset(days=2)
    if mat_date is None:
        mat_date = eff_date + _pd.DateOffset(years=2)
    if float_curve is None:
        float_curve = us_swap
    if disc_curve is None:
        disc_curve = us_swap

    dates = custom_date_range(eff_date, mat_date, freq=reset_freq)

    # in case mat_date does not fall evenly on freq, take last date before
    dates = dates[dates <= mat_date]
    dates = _pd.Index([_pd.to_datetime(trade_date)]).append(dates)

    if (days_in_year in [360, 365]) == False:
        raise ValueError("days_in_year must be either 360 or 365")

    # placeholder
    if convention != "act":
        raise ValueError("function only defined for convention='act'")

    df = _pd.DataFrame(
        {
            "dates": dates,
            "day2next": (dates[1:] - dates[:-1]).days.append(
                _pd.Index([0])
            ),  # calc days to next period, short one element at end so add zero
            "times": (dates - dates[0]).days
            / 365,  # calc days to maturity from trade_date
        }
    )
    print(df)
    disc = _interpolate.splrep(disc_curve["times"], disc_curve["discounts"])
    df["disc"] = _interpolate.splev(df.times, disc)

    df["fixed"] = notional * fixed_rate * (df.day2next / days_in_year)
    df.loc[df.day2next <= 20, "fixed"] = 0
    df.fixed = df.fixed.shift() * df.disc

    disc_float = _interpolate.splrep(float_curve["times"], float_curve["discounts"])
    df["disc_float"] = _interpolate.splev(df.times, disc_float)
    df["floating"] = notional * (df.disc_float / df.disc_float.shift(-1) - 1)
    df.loc[df.day2next <= 20, "floating"] = 0
    df.floating = df.floating.shift() * df.disc
    df["net"] = df.fixed - df.floating

    df = df.fillna(0)
    pv = df.net.sum()
    df["duration"] = (
        (dates - _pd.to_datetime(trade_date)).days / days_in_year * df.net / pv
    )
    duration = df.duration.sum()

    if pay_rec == "pay":
        pv *= -1

    if output == "price":
        return pv
    else:
        return {"pv": pv, "df": df, "duration": duration}


def swap_com(df, futures_names, start_dt, end_dt, cmdty, exchange):
    """
    Function to calculate commodity swap pricing from futures contract exchange settlements of two contracts.
    Mostly used when the contract expires mid-month to work out the calendar month average

    Parameters
    ----------
    df : DataFrame
        Wide dataframe of futures prices
    futures_names : List[Tuple[str]]
        2 element List or tuple of futures contract names in order of expiry (i.e. [first expirying contract, second expirying contract])
    start_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    end_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    cmdty : str
        name of expiry contract type as specified in the expiry_table of this package. See risktools.data.open_data('expiry_table')['cmdty']
    exchange : ['nymex','ice']
        Name of exchange. Only 'nymex' and 'ice' supported. Used to filter the dataframe risktools.data.open_data('holidaysOil')

    Returns
    -------
    a dataframe of swap prices

    Examples
    --------
    >>> df = rt.get_prices(up['m*']['user'], up['m*']['pass'], codes=['CL0M','CL0N','CL0Q'], start_dt='2019-08-26')
    >>> df = df.settlement_price.unstack(level=0)
    >>> rt.swap_com(df=df, futures_names=["CL0M","CL0N"], start_dt="2020-05-01", end_dt="2020-05-30", cmdty="cmewti", exchange="nymex")
    """
    # get contract expiry dates
    exp = data.open_data("expiry_table")
    exp = exp.query(
        f"Last_Trade >= '{start_dt}' & Last_Trade <= '{end_dt}' & cmdty == '{cmdty}'"
    )["Last_Trade"]

    hol = data.open_data("holidaysOil").query(f"key == '{exchange}'")

    biz_dates = set(_pd.date_range(start_dt, end_dt, freq="B").to_list())
    biz_dates = biz_dates - set(hol.value)

    res = _pd.DataFrame(index=biz_dates).sort_index()
    res["up2expiry"] = 1

    next_contract_dt = _pd.to_datetime(exp.values[0]) + _pd.DateOffset(days=1)

    res.loc[next_contract_dt:, "up2expiry"] = 0

    first_fut_weight = res.up2expiry.sum() / res.up2expiry.count()

    df = df[futures_names]
    df["swap"] = df.iloc[:, 0] * first_fut_weight + df.iloc[:, 1] * (
        1 - first_fut_weight
    )

    return df.dropna()


def get_ir_swap_curve(
    username, password, currency="USD", start_dt="2019-01-01", end_dt=None
):
    """
    Extract historical interest rate swap data for Quantlib DiscountsCurve function
    using Morningstar and FRED data

    Parameters
    ----------

    username
    password
    currency
    start_dt
    end_dt

    Examples
    --------
    """

    # fmt: off
    libor_ticks = _pd.DataFrame(dict(
        tick_nm = ["d1d","d1w","d1m","d3m","d6m","d1y"],
        type = ["ICE.LIBOR"]*6,
        source = ["FRED"]*6,
        codes = ["USDONTD156N","USD1WKD156N","USD1MTD156N","USD3MTD156N","USD6MTD156N","USD12MD156N"]
    ))

    mstar_ticks = _pd.DataFrame(dict(
        tick_nm = ["fut" + str(i) for i in range(1,9)],
        type = ["EuroDollar"]*8,
        source = ["Morningstar"]*8,
        codes = [f"ED_{str(i).zfill(3)}_Month" for i in range(1,9)]
    ))

    yrs = [2,3,5,7,10,15,20,30]
    irs_ticks = _pd.DataFrame(dict(
        tick_nm = [f"s{str(i)}y" for i in yrs],
        type = ["IRS"]*8,
        source = ["FRED"]*8,
        codes = [f"ICERATES1100USD{str(i)}Y" for i in yrs]
    ))

    ticks = libor_ticks.append(irs_ticks).append(mstar_ticks)
    # fmt: on

    r = get_prices(
        codes=ticks.loc[ticks.source == "Morningstar", "codes"].to_list(),
        feed="CME_CmeFutures_EOD_continuous",
        username=username,
        password=password,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    f = _pdr.DataReader(
        name=ticks.loc[ticks.source == "FRED", "codes"].to_list(),
        data_source="fred",
        start=start_dt,
        end=end_dt,
    )

    f.index.name = "Date"
    f.index = _pd.to_datetime(f.index)
    df = (
        r.unstack(0)
        .droplevel(0, 1)
        .merge(f, left_index=True, right_index=True, how="outer")
    )

    df[df.columns[df.columns.str.contains("ICERATES")]] /= 100
    df[df.columns[df.columns.str[:3] == "USD"]] /= 100
    df = df.rename(ticks.set_index("codes").tick_nm.to_dict(), axis=1)
    df = df[ticks.tick_nm]

    return df


def swap_info(
    username,
    password,
    date=None,
    tickers=["CL", "CL_001_Month"],
    feeds=[
        "Crb_Futures_Price_Volume_And_Open_Interest",
        "CME_NymexFutures_EOD_continuous",
    ],
    exchange="nymex",
    contract="cmewti",
    output="all",
):
    """
    Returns dataframe required to price a WTI averaging instrument based on first line settlements.

    Parameters
    ----------
    username : str
        Morningstar user name as character - sourced locally in examples.
    password : str
        Morningstar user password as character - sourced locally in examples.
    date : str | datetime
        Date as of which you want to extract daily settlement and forward values. By default None
    tickers : str | iterable[str]
        Morningstar tickers for get_curve() and get_prices() functions
    feeds : str | iterable[str]
        Feeds for Morningstar get_curve() and get_prices() functions. Must be in same 
        order as tickers parameter
    exchange : str
        Exchange code in holidaysOil from risktools.data.open_data function. Defaults to "nymex".
    contract : str
        Contract code in expiry_table.  from risktools.data.open_data function. See
        risktools.data.open_date("expiry_table").cmdty.unique() for options.
    output :str
        "chart" for chart, "dataframe" for dataframe or "all" dataframe and chart

    Returns
    -------
    Plot, dataframe or a dict of data frame and plot if output = "all".

    Examples
    --------
    >>> import risktools as rt
    >>> feeds = ["Crb_Futures_Price_Volume_And_Open_Interest",
            "CME_NymexFutures_EOD_continuous"]
    >>> ticker = ["CL","CL_001_Month"]
    >>> swap_info(username, password, date="2020-05-06", tickers=tickers, feeds=feeds, contract="cmewti", exchange="nymex",
            output = "all")
    """
    # get today's futures curve prices by expiration date
    wti = get_curves(
        username=username,
        password=password,
        feed=feeds[0],
        contract_roots=tickers[0],
        fields=["Close"],
        date=date,
    )

    if date is None:
        date = _pd.Timestamp.now().floor("D")
    else:
        date = _pd.to_datetime(date)

    # broadcast wti expiration date to daily prices (effectively a stepped curve)
    drng = _pd.date_range(start=date, end=date + _pd.DateOffset(months=4), freq="B")

    cal = data.open_data("holidaysOil")
    cal = cal[cal.key == exchange]

    df = _pd.DataFrame(index=drng)
    # remove nymex holidays
    df = df[~df.index.isin(cal.value)]

    # combine dataframes together by broadcasting wti to wf on expiration date
    # and then backfill
    df.index.name = "date"
    wti = wti.rename({"expirationDate": "date"}, axis=1)
    df = df.join(wti.set_index("date")).bfill()

    hist = get_prices(
        username=username,
        password=password,
        feed=feeds[1],
        codes=tickers[1],
        start_dt=date + _pd.DateOffset(day=1),
    )

    hist = (
        hist.reset_index()[["Date", "settlement_price"]]
        .rename({"Date": "date", "settlement_price": "price"}, axis=1)
        .set_index("date")
    )
    hist["futures_contract"] = "first_line_settled"
    df = df.rename({"code": "futures_contract", "Close": "price"}, axis=1)[
        ["futures_contract", "price"]
    ]
    df = (
        hist.loc[:(date), :]
        .append(df.loc[(date + _pd.DateOffset(days=1)) :, :])
        .dropna()
    )

    if output == "dataframe":
        out = df
    else:
        title = (
            f"Swap Pricing: {contract}, Futures Curve as of {date.strftime('%Y-%m-%d')}"
        )

        fig = _px.scatter(
            df,
            y="price",
            color="futures_contract",
            title=title,
            labels={"price": "$/bbl"},
        )
        if output == "chart":
            out = fig
        else:
            out = dict(chart=fig, dataframe=df)

    return out


def swap_fut_weight(
    month, contract="cmewti", exchange="nymex", output="first_fut_weight"
):
    """
    Function used to calculate the calendar month average price of a futures curve. Some 
    futures curves such as NYMEX WTI expires mid-month (i.e. 2020-09-22) but some products
    are priced as differentials to the calendar month average (CMA) price of the contracts. 
    To calculate the CMA, up need the contract that's active for the first part of the month, 
    the contract that's active for the later half as well as the weighting to apply to both. 
    This function is designed to return the weights for contracts 1 and 2 as well as the weights

    Parameters
    ----------
    month : str | datetime
        First calendar day of the month in YYYY-MM-DD
    contract : str
        Exchange code in data.open_data("holidaysOil"). Currently only "nymex" and "ice" supported.
        By default "cmewti"
    exchange : str
        Contract code in data.open_data("expiry_table"). By default "nymex"
    output : str
        "num_days_fut1" to get the number of days from the beginning of the month up to and including the expiry date.
        "num_days_fut2" to get the number of days from the expiry date to month end.
        "first_fut_weight" the weight to assign to the futures contract for "month" as a ratio of the number of days before expiry (i.e. include in ). By default "first_fut_weight"

    Returns
    -------
    If first_fut_weight as dataframe, if not then an integer
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.swap_fut_weight(month="2020-09-01", contract="cmewti", exchange="nymex", output="first_fut_weight")
    """

    month = _pd.to_datetime(month) + _pd.DateOffset(day=1)
    month_end = month + _pd.DateOffset(months=1) - _pd.DateOffset(days=1)

    drng = _pd.date_range(month, month_end, freq="B")

    cal = data.open_data("holidaysOil")
    cal = cal[cal.key == exchange]

    df = _pd.DataFrame(index=drng)
    # remove nymex holidays
    df = df[~df.index.isin(cal.value)]

    exp = data.open_data("expiry_table")
    exp = exp[
        (exp["Last_Trade"] >= month)
        & (exp["Last_Trade"] <= month_end)
        & (exp["cmdty"] == contract)
    ].Last_Trade.values[0]

    df["up_to_expiry"] = _np.where(df.index <= exp, 1, 0)
    num_days_fut1 = df.up_to_expiry.sum()
    num_days_fut2 = df.shape[0] - df.up_to_expiry.sum()
    first_fut_weight = num_days_fut1 / df.shape[0]

    if output == "num_days_fut1":
        out = num_days_fut1
    elif output == "num_days_fut2":
        out = num_days_fut2
    else:
        out = first_fut_weight

    return out


# def swap_irs(
#     trade_dt=None,
#     eff_dt=None,
#     mat_dt=None,
#     notional=1e6,
#     pay=False,
#     fixed_rate=0.05,
#     float_curve=None,
#     reset_freq=3,
#     disc_curve=None,
#     convention=("act", 360),
#     bus_calendar="NY",
#     outout="price",
# ):

#     """
#     Commodity swap pricing from exchange settlement

#     Parameters
#     ----------

#     trade_dt : datetime | str
#         trade date as a string or datetime object. By deault None which makes the trade date today.
#     eff_dt : datetime | str
#         effective date of swap as a string or datetime object. By deault None which makes the effective
#         date today + 2 days.
#     mat_dt : datetime | str
#         maturity date of swap as a string or datetime object. By deault None which makes the maturity
#         date the effective date + 2 years.
#     notional : nmumeric
#         Numeric value of notional. Defaults to 1,000,000
#     pay : bool
#         True for "pay" and False for "receive"
#     fixed_rate : float
#         Numeric fixed interest rate. Defaults to 0.05.
#     float_curve : DataFrame
#         List of interest rate curves. Defaults to data("usSwapCurves").
#     reset_freq : int
#         Numeric where 1 = "monthly", 3 = quarterly, 6 = Semi annual 12 = yearly.
#     disc_curve : DataFrame
#         List of interest rate curves. Defaults to data("usSwapCurves").
#     convention : tuple(str|numeric, numeric)
#         Tuple of convention e.g. ("act",360), (30,360),...
#         Tuple definition (, days in year)
#     bus_calendar : str
#         Banking day calendar. Not implemented.
#     output : str
#         "price" for swap price or "all" for price, cash flow data frame, duration.

#     Returns
#     -------
#     Dictionary with swap price, cash flow dataframe and duration

#     Example
#     -------
#     >>> import risktools as rt
#     >>> us_swap = rt.data.open_data("usSwapCurves")
#     >>> rt.swap_irs(trade_dt="2020-01-04", eff_dt="2020-01-06",
#             mat_dt="2022-01-06", notional=1000000,
#             pay=False, fixed_rate=0.05, float_curve = us_swap, reset_freq=3,
#             disc_curve = us_swap, convention = ("act",360),
#             bus_calendar = "NY", output = "all")
#     """

#     if trade_dt is None:
#         trade_dt = _pd.Timestamp.now().floor("D")
#     else:
#         trade_dt = _pd.to_datetime(trade_dt)
#     if eff_dt is None:
#         eff_dt = trade_dt + _pd.DateOffset(days=2)
#     else:
#         eff_dt = _pd.to_datetime(eff_dt)
#     if mat_dt is None:
#         mat_dt = eff_dt + _pd.DateOffset(years=2)
#     else:
#         mat_dt = _pd.to_datetime(mat_dt)

#     dates = _pd.date_range(
#         eff_dt,
#         mat_dt,
#         freq=_pd.DateOffset(months=reset_freq, day=eff_dt.day),
#         closed="left",
#     )

#     dates = _pd.date_range(trade_dt, end=trade_dt).append(dates)
#     dates = _pd.DataFrame(dict(dates=dates))["dates"]

#     # return dates

#     days_in_year = int(convention[1])

#     if days_in_year not in (360, 365):
#         raise ValueError("# days in year convention not defined")

#     if convention[0] != "act":
#         raise ValueError("function only defined for act")

#     tf = _pd.DataFrame(
#         dict(
#             dates=dates,
#             days_to_next=(dates.shift(-1) - dates).dt.days.fillna(0).astype(int),
#             times=(dates - trade_dt).astype("timedelta64[h]") / days_in_year / 24,
#         )
#     )

#     return tf

