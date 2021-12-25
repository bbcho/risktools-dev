from . import data

import pandas as pd
from scipy import interpolate
from ._morningstar import *

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

    dr = [pd.to_datetime(start)]

    i = 1
    while dr[-1] < pd.to_datetime(end):
        dr.append(dr[0] + pd.DateOffset(months=mul * i))
        i += 1

    return pd.Index(dr)


def swap_irs(
    trade_date=pd.Timestamp.now().floor("D"),
    eff_date=pd.Timestamp.now().floor("D") + pd.DateOffset(days=2),
    mat_date=pd.Timestamp.now().floor("D")
    + pd.DateOffset(days=2)
    + pd.DateOffset(years=2),
    notional=1000000,
    pay_rec="rec",
    fixed_rate=0.05,
    float_curve=us_swap,
    reset_freq="Q",
    disc_curve=us_swap,
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
        Defaults to today().
    eff_date : Timestamp | str
        Defaults to today() + 2 days.
    mat_date : Timestamp | str
        Defaults to today() + 2 years.
    notional : long int
        Numeric value of notional. Defaults to 1,000,000.
    pay_rec : str
        "pay" for pyement (positive pv) or "rec" for receivables (negative pv).
    fixed_rate : float
        fixed interest rate. Defaults to 0.05.
    float_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. Defaults to rt.data.open_data("usSwapCurves")
        which is a dictionary based on the R object DiscountCurve. Column times is year fractions going forward in time.
        So today is 0, tomorrow is 1/365, a year from today is 1 etc...
    reset_freq : str
        Pandas/datetime Timestamp frequency (allowable values are 'M', 'Q', '6M', or 'Y')
    disc_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. Defaults to rt.data.open_data("usSwapCurves")
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
    dates = custom_date_range(eff_date, mat_date, freq=reset_freq)

    # in case mat_date does not fall evenly on freq, take last date before
    dates = dates[dates <= mat_date]
    dates = pd.Index([pd.to_datetime(trade_date)]).append(dates)

    if (days_in_year in [360, 365]) == False:
        raise ValueError("days_in_year must be either 360 or 365")

    # placeholder
    if convention != "act":
        raise ValueError("function only defined for convention='act'")

    df = pd.DataFrame(
        {
            "dates": dates,
            "day2next": (dates[1:] - dates[:-1]).days.append(
                pd.Index([0])
            ),  # calc days to next period, short one element at end so add zero
            "times": (dates - dates[0]).days
            / 365,  # calc days to maturity from trade_date
        }
    )
    print(df)
    disc = interpolate.splrep(disc_curve["times"], disc_curve["discounts"])
    df["disc"] = interpolate.splev(df.times, disc)

    df["fixed"] = notional * fixed_rate * (df.day2next / days_in_year)
    df.loc[df.day2next <= 20, "fixed"] = 0
    df.fixed = df.fixed.shift() * df.disc

    disc_float = interpolate.splrep(float_curve["times"], float_curve["discounts"])
    df["disc_float"] = interpolate.splev(df.times, disc_float)
    df["floating"] = notional * (df.disc_float / df.disc_float.shift(-1) - 1)
    df.loc[df.day2next <= 20, "floating"] = 0
    df.floating = df.floating.shift() * df.disc
    df["net"] = df.fixed - df.floating

    df = df.fillna(0)
    pv = df.net.sum()
    df["duration"] = (
        (dates - pd.to_datetime(trade_date)).days / days_in_year * df.net / pv
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

    biz_dates = set(pd.date_range(start_dt, end_dt, freq="B").to_list())
    biz_dates = biz_dates - set(hol.value)

    res = pd.DataFrame(index=biz_dates).sort_index()
    res["up2expiry"] = 1

    next_contract_dt = pd.to_datetime(exp.values[0]) + pd.DateOffset(days=1)

    res.loc[next_contract_dt:, "up2expiry"] = 0

    first_fut_weight = res.up2expiry.sum() / res.up2expiry.count()

    df = df[futures_names]
    df["swap"] = df.iloc[:, 0] * first_fut_weight + df.iloc[:, 1] * (
        1 - first_fut_weight
    )

    return df.dropna()
