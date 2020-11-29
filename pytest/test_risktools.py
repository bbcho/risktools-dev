import pandas as pd
import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
from pandas_datareader import data

with open("../../user.json") as js:
    up = json.load(js)


def _load_json(fn, dataframe=True):
    path = os.path.dirname(__file__)
    fp = os.path.join(path, fn)
    with open(fp) as js:
        if dataframe == True:
            df = pd.read_json(js)
            df.columns = df.columns.str.replace(".", "_")
        else:
            df = json.loads(fp)

    return df


def test_ir_df_us():
    df = _load_json("ir_df_us.json")
    ir = rt.ir_df_us(quandl_key=up["quandl"])
    ir = ir[
        [
            "yield",
            "maturity",
            "discountfactor",
            "discountfactor_plus",
            "discountfactor_minus",
        ]
    ]
    assert df.round(4).equals(
        ir.round(4)
    ), "ir_df_us test failed, returned dataframe does not equal RTL results"


def test_bond():
    # first test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="price")
    assert bo == 100, "bond Test 1 failed"

    # second test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="df")
    df = _load_json("bond_2.json")
    assert df.astype(float).round(4).equals(bo.round(4)), "bond Test 2 failed"

    # third test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="duration")
    assert round(bo, 4) == 0.9878, "bond Test 3 failed"


def test_trade_stats():

    df = data.DataReader(["SPY", "AAPL"], "yahoo", "2000-01-01", "2012-01-01")
    df = df.pct_change()
    df = df.asfreq("B")

    ou = rt.trade_stats(df[("Adj Close", "SPY")])
    ts = _load_json("tradeStats.json")

    assert round(ou["cum_ret"], 4) == round(
        ts["CumReturn"][0], 4
    ), "tradeStats Test cum_ret failed"
    assert round(ou["ret_ann"], 4) == round(
        ts["Ret_Ann"][0], 4
    ), "tradeStats Test ret_ann failed"
    assert round(ou["sd_ann"], 4) == round(
        ts["SD_Ann"][0], 4
    ), "tradeStats Test sd_ann failed"
    assert round(ou["omega"], 4) == round(
        ts["Omega"][0], 4
    ), "tradeStats Test omega failed"
    assert round(ou["sharpe"], 4) == round(
        ts["Sharpe"][0], 4
    ), "tradeStats Test sharpe failed"
    assert round(ou["perc_win"], 4) == round(
        ts["%_Win"][0], 4
    ), "tradeStats Test perc_win failed"
    assert round(ou["perc_in_mkt"], 4) == round(
        ts["%_InMrkt"][0], 4
    ), "tradeStats Test perc_in_mkt failed"
    assert round(ou["dd_length"], 4) == round(
        ts["DD_Length"][0], 4
    ), "tradeStats Test dd_length failed"
    assert round(ou["dd_max"], 4) == round(
        ts["DD_Max"][0], 4
    ), "tradeStats Test dd_max failed"


def test_returns():

    # Test 1
    ac1 = _load_json("returns1.json").round(4).set_index("date").dropna()
    ac1.columns.name = "series"
    ts1 = (
        rt.returns(
            df=rt.data.open_data("dflong").round(
                4
            ),  # round(4) because R toJSON function does so
            ret_type="rel",
            period_return=1,
            spread=True,
        )
        .round(4)
        .iloc[1:, :]  # for some reason the RTL version drops the first row
    )
    assert ac1.equals(ts1), "returns Test 1 failed"

    # Test 2
    ac2 = _load_json("returns2.json").round(4)

    ts2 = rt.returns(
        df=rt.data.open_data("dflong").round(4),
        ret_type="rel",
        period_return=1,
        spread=False,
    )
    ts2 = ts2.round(4)

    ac2 = ac2.set_index(["series", "date"])["returns"].sort_index()

    # remove first date. RTL does this for some reason
    ts2 = ts2.unstack(0).iloc[1:, :].stack().swaplevel(0, 1).sort_index()

    assert ac2.equals(ts2), "returns Test 2 failed"

    # Test 3
    ac = _load_json("returns3.json").round(4).set_index("date").dropna()
    ac.columns.name = "series"
    ts = (
        rt.returns(
            df=rt.data.open_data("dflong").round(
                4
            ),  # round(4) because R toJSON function does so
            ret_type="abs",
            period_return=1,
            spread=True,
        )
        .round(4)
        .iloc[1:, :]  # for some reason the RTL version drops the first row
    )
    assert ac.equals(ts), "returns Test 3 failed"

    # Test 4
    ac = _load_json("returns4.json").round(4).set_index("date").dropna()
    ac.columns.name = "series"
    ts = (
        rt.returns(
            df=rt.data.open_data("dflong").round(
                4
            ),  # round(4) because R toJSON function does so
            ret_type="log",
            period_return=1,
            spread=True,
        )
        .round(4)
        .iloc[1:, :]  # for some reason the RTL version drops the first row
    )
    assert ac.equals(ts), "returns Test 4 failed"




if __name__ == "__main__":
    test_returns()
    pass


# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri, numpy2ri
# from rpy2.robjects.pandas2ri import rpy2py, py2rpy
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.packages import importr
# from rpy2.robjects.packages import STAP
# from contextlib import contextmanager

# # R vector of strings
# from rpy2.robjects.vectors import StrVector, FloatVector, ListVector, Vector

# # import R's "base" package
# base = importr("base")

# utils = importr("utils")
# tq = importr("tidyquant")
# tv = importr("tidyverse")
# ql = importr("Quandl")
# rtl = importr("RTL")


# def p2r(p_df):
#     # Function to convert pandas dataframes to R
#     with localconverter(ro.default_converter + pandas2ri.converter) as cv:
#         r_from_pd_df = cv.py2rpy(p_df)

#     return r_from_pd_df


# def r2p(r_df):
#     # Function to convert R dataframes to pandas
#     with localconverter(ro.default_converter + pandas2ri.converter) as cv:
#         pd_from_r_df = cv.rpy2py(r_df)

#     return pd_from_r_df
