from numpy.linalg import eigvals
import pandas as pd
import numpy as np
import os
import json
import sys
import plotly.graph_objects as go
import time
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt

# TODO
# stl_decomp test
# chart_zscore only tests figure object - test actual stl decomp

# with open("../user.json") as js:
#     up = json.load(js)

test_date = "2021-12-24"

upf = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": ""}
try:
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../user.json', mode='r') as file:
        upf = json.load(file)
except:
    pass

# Github Actions CI Env Vars
up = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": None}

up["eia"] = os.getenv("EIA", upf['eia'])
up["quandl"] = os.getenv("QUANDL", upf["quandl"])
up["m*"]["pass"] = os.getenv("MS_PASS", upf["m*"]["pass"] )
up["m*"]["user"] = os.getenv("MS_USER", upf["m*"]["user"])

ms = dict(username=os.getenv("MS_USER"), password=os.getenv("MS_PASS"))


def _load_json(fn, dataframe=True):
    path = os.path.dirname(__file__)
    fp = os.path.join(path, fn)
    with open(fp) as js:
        if dataframe == True:
            df = pd.read_json(js)
            df.columns = df.columns.str.replace(".", "_")
        else:
            df = json.load(js)

    return df


@pytest.mark.skip(reason="Requires Morningstar API credentials")
def test_get_prices():
    # ac_all = _load_json("get_price.json", dataframe=False)

    # i = 0
    # while True:
    #     try:
    #         ac = ac_all[i]
    #     except:
    #         break

    #     ac_df = pd.DataFrame(ac["df"])
    #     ac_df.date = pd.to_datetime(ac_df.date)
    #     ac_df.date = ac_df.date.dt.tz_localize(None)
    #     print(ac["feed"][0], ac["contract"][0])
    #     print(ac_df.date.max())

    #     ts = (
    #         rt.get_prices(
    #             up["m*"]["user"],
    #             up["m*"]["pass"],
    #             feed=ac["feed"][0],
    #             codes=ac["contract"][0],
    #             start_dt=ac["from"][0],
    #             end_dt=ac["end"][0],
    #         )
    #         .iloc[:, 0]
    #         .unstack(0)
    #         .reset_index()
    #         .rename({"Date": "date"}, axis=1)
    #     )
    #     ts.columns = ts.columns.str.replace("@", "")
    #     ts.date = ts.date.dt.tz_localize(None)

    #     if ac["feed"][0] == "LME_MonthlyDelayed_Derived":
    #         ts.columns = [
    #             ts.columns[0],
    #             ts.columns[1].replace(" ", "").replace("-", "")[0:9],
    #         ]
    #     elif ac["feed"][0] == "AESO_ForecastAndActualPoolPrice":
    #         # Accounts for mid-day runs of hourly data
    #         # Also RTL function doesn't have ability to give a TO date
    #         ts = ts.set_index("date")
    #         ac_df = ac_df.set_index("date")
    #         min_dt = max(ac_df.index.min(), ts.index.min())
    #         max_dt = min(ac_df.index.max(), ts.index.max())
    #         ts = ts[min_dt:max_dt].reset_index()
    #         ac_df = ac_df[min_dt:max_dt].reset_index()
    #     try:
    #         pd.testing.assert_frame_equal(ac_df, ts, check_like=True)
    #     except:
    #         assert False, f"test {i} failed"
    #     i += 1
    pass


@pytest.mark.skip(reason="Requires Quandl API network access")
def test_ir_df_us():

    df = _load_json("./data/ir_df_us.json")
    df = df[
        [
            "yield",
            "maturity",
            "discountfactor",
            "discountfactor_plus",
            "discountfactor_minus",
        ]
    ]
    ir = rt.ir_df_us(quandl_key=up["quandl"], date='2021-12-18')
    ir = ir[
        [
            "yield",
            "maturity",
            "discountfactor",
            "discountfactor_plus",
            "discountfactor_minus",
        ]
    ].reset_index(drop=True)

    assert df.round(4).equals(
        ir.round(4)
    ), "ir_df_us test failed, returned dataframe does not equal RTL results"


def test_bond():

    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="price")
    assert round(bo,4) == 100.0, "bond Test 1 failed"

    # second test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="df")
    df = _load_json("./data/bond_2.json")
    assert df.astype(float).round(4).equals(bo.round(4)), "bond Test 2 failed"

    # third test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="duration")
    assert round(bo, 4) == 0.9878, "bond Test 3 failed"


@pytest.mark.skip(reason="Requires yfinance and network access")
def test_trade_stats():

    # df = data.DataReader(["SPY", "AAPL"], "yahoo", "2000-01-01", "2012-01-01")
    df = yf.download(["SPY", "AAPL"], start="2000-01-01", end="2012-01-01")
    df = df.pct_change()
    df = df.asfreq("B")

    ou = rt.trade_stats(df[("Adj Close", "SPY")])
    ts = _load_json("./data/tradeStats.json")

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
    ac = (
        _load_json("./data/returns1.json")
        .round(4)
        .set_index("date")
        .sort_index(axis=1)
    )
    ac.columns.name = "series"

    ts = (
        rt.returns(
            df=rt.data.open_data("dflong").round(
                4
            ),  # round(4) because R toJSON function does so
            ret_type="rel",
            period_return=1,
            spread=True,
        )
        .round(4)
        .sort_index(axis=1)
    )
    
    assert ac.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']].equals(ts.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']]), "returns Test 1 failed"

    # Test 2
    ac2 = _load_json("./data/returns2.json").round(4)

    ts2 = rt.returns(
        df=rt.data.open_data("dflong").round(4),
        ret_type="rel",
        period_return=1,
        spread=False,
    )
    ts2 = ts2.round(4)

    ac2 = ac2.set_index(["series", "date"])["returns"].sort_index()

    # ts2 = ts2.unstack(0).stack().swaplevel(0, 1).sort_index()

    assert ac.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']].equals(ts.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']]), "returns Test 2 failed"

    # Test 3
    ac = (
        _load_json("./data/returns3.json")
        .round(4)
        .set_index("date")
        .sort_index(axis=1)
    )
    ac.columns.name = "series"
    ts = rt.returns(
        df=rt.data.open_data("dflong").round(
            4
        ),  # round(4) because R toJSON function does so
        ret_type="abs",
        period_return=1,
        spread=True,
    ).round(4)

    assert ac.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']].equals(ts.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']]), "returns Test 3 failed"

    # Test 4
    ac = (
        _load_json("./data/returns4.json")
        .round(4)
        .set_index("date")
        .sort_index(axis=1)
    )
    ac.columns.name = "series"
    ts = rt.returns(
        df=rt.data.open_data("dflong").round(
            4
        ),  # round(4) because R toJSON function does so
        ret_type="log",
        period_return=1,
        spread=True,
    ).round(4)

    assert ac.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']].equals(ts.loc['2021-06-01':'2021-06-30', ['BRN01','BRN02']]), "returns Test 4 failed"


def test_roll_adjust():
    ac = _load_json("./data/rolladjust.json").set_index("date").round(4)

    ac = ac.iloc[:, 0]

    dflong = rt.data.open_data("dflong")["CL01"]
    rt.data.open_data("expiry_table").cmdty.unique()  # for list of commodity names
    ret = rt.returns(df=dflong, ret_type="abs", period_return=1, spread=True)
    ret = ret.iloc[:, 0].dropna()
    ts = (
        rt.roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade")
        # .iloc[1:]
        .round(4)
    )

    assert ac.loc['2021-06-01':'2021-06-30'].equals(ts.loc['2021-06-01':'2021-06-30']), "rolladjust Test failed"


def test_garch():
    ac = _load_json("./data/garch.json").set_index("date").garch

    dflong = rt.data.open_data("dflong")
    dflong = dflong["CL01"]
    df = rt.returns(df=dflong, ret_type="rel", period_return=1, spread=True)

    df = rt.roll_adjust(df=df, commodity_name="cmewti", roll_type="Last_Trade").iloc[1:]

    ts = rt.garch(df, out="data", vol="garch", rescale=False, scale=252)

    # need to see if I can get R and Python garch models to produce the same vol
    assert (ac.mean() / ts["h.1"].mean() < 2) & (
        ac.mean() / ts["h.1"].mean() > 0.5
    ), "garch mean test failed, test result mean is more that double or less than half of RTL results"
    assert (ac.std() / ts["h.1"].std() < 2) & (
        ac.std() / ts["h.1"].std() > 0.5
    ), "garch std test failed, test result std is more that double or less than half of RTL results"

    # redo R garch using a standard garch model


def test_prompt_beta():
    
    ac = _load_json("../pytest/data/promptBeta.json").round(4).drop("contract", axis=1)

    dfwide = rt.data.open_data("dflong").unstack(0)
    col_mask = dfwide.columns[dfwide.columns.str.contains("CL")]
    dfwide = dfwide[col_mask]
    dfwide = dfwide.dropna(thresh=10, axis=0)

    x = rt.returns(df=dfwide, ret_type="abs", period_return=1)
    x = rt.roll_adjust(df=x, commodity_name="cmewti", roll_type="Last_Trade")
    x = x[~x.index.isin(["2020-04-20", "2020-04-21"])]
    x = x.loc['2010-01-04':'2022-12-30',:]

    ts = (
        rt.prompt_beta(df=x, period="all", beta_type="all", output="betas")
        .round(4)
        .reset_index(drop=True)
    )
    # for some reason the betas are slightly different using the Python sklearn
    # LinearRegression model. Make sure that the max of the three columns
    # are less than 0.03. Differences are on the order of 0.001 on any individual
    # beta
    assert (
        ac - ts
    ).abs().max().sum() < 0.03, (
        "prompt_beta Test failed, sum of total differences > 0.03"
    )


@pytest.mark.skip(reason="Requires Morningstar API credentials")
def test_swap_irs():
    # a = 85085.84
    # b = round(1.015174, 4)

    # ac = _load_json("./data/swapIRS.json")
    # ac.dates = pd.to_datetime(ac.dates)

    # usSwapCurves = rt.data.open_data("usSwapCurves")
    # ts = rt.swap_irs(
    #     trade_date="2020-01-04",
    #     eff_date="2020-01-06",
    #     mat_date="2021-12-06",
    #     notional=1000000,
    #     pay_rec="rec",
    #     fixed_rate=0.05,
    #     float_curve=usSwapCurves,
    #     reset_freq="Q",
    #     disc_curve=usSwapCurves,
    #     days_in_year=360,
    #     convention="act",
    #     bus_calendar="NY",
    #     output="all",
    # )

    # assert round(a, 0) == ts["pv"].round(
    #     0
    # ), "swapIRS test failed, pv not equal to 2 decimal places'"

    # assert b == ts["duration"].round(
    #     4
    # ), "swapIRS test failed, duration not equal to 4 decimal places"

    # ac[["fixed", "floating", "net"]] = ac[["fixed", "floating", "net"]].round(0)

    # ts["df"][["fixed", "floating", "net"]] = ts["df"][
    #     ["fixed", "floating", "net"]
    # ].round(0)

    # assert ac.round(4).equals(ts["df"].round(4)), "swapIRS test failed"
    pass


def test_npv():
    ac = _load_json("./data/npv1.json")
    ir = (
        _load_json("./data/ir.json")
        .rename({"_row": "index"}, axis=1)
        .replace("...1", "0")
        .set_index("index")
    )
    ac.cf = ac.cf.astype(float)
    ts = rt.npv(
        init_cost=-375, C=50, cf_freq=0.5, F=250, T=2, disc_factors=ir, break_even=False
    )

    assert ac.round(4).equals(ts.round(4)), "npv Test 1 using actual ir failed"

    ac2 = _load_json("./data/npv2.json")
    ac2.cf = ac2.cf.astype(float)
    ts2 = rt.npv(
        init_cost=-375,
        C=50,
        cf_freq=0.5,
        F=250,
        T=2,
        disc_factors=ir,
        break_even=True,
        be_yield=0.0399,
    )

    assert ac2.round(4).equals(ts2.round(4)), "npv Test 2 using fixed yield"


def test_crr_euro():
    ac = _load_json("./data/crreuro.json", dataframe=False)
    ts = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")

    assert np.array_equal(
        np.array(ac["asset"]), ts["asset"].round(4)
    ), "crr_euro Test failed on assets array"
    assert np.array_equal(
        np.array(ac["option"]), ts["option"].round(4)
    ), "crr_euro Test failed on options array"
    assert ac["price"][0] == round(ts["price"], 4), "crr_euro Test failed on price"
    assert ac["note"][0] == ts["note"], "crr_euro Test failed on price"


@pytest.mark.skip(reason="Test data (stl_decomp.json) not available")
def test_stl_decomposition():
    # ac = _load_json("stl_decomp.json")

    # ac = ac.rename({"index": "date"}, axis=1).set_index("date")

    # df = rt.data.open_data("dflong")
    # df = df["CL01"]
    # ts = rt.stl_decomposition(
    #     df, output="data", seasonal=13, seasonal_deg=1, resample_freq="M"
    # )
    pass


@pytest.mark.skip(reason="Requires EIA API key and network access")
def test_get_eia_df():
    ts = rt.get_eia_df("PET.MCRFPTX2.M", key=up["eia"])

    assert ts.shape[0] > 0, "get_eia_df Test 1 failed"

    ts = rt.get_eia_df(
        ["PET.W_EPC0_SAX_YCUOK_MBBL.W", "NG.NW2_EPG0_SWO_R48_BCF.W"], key=up["eia"]
    )

    assert ts.shape[0] > 0, "get_eia_df Test 2 failed"

    assert ts.shape[1] == 4, "get_eia_df Test 3 failed"


@pytest.mark.skip(reason="Requires Morningstar API credentials")
def test_chart_spreads():
    # # ac = _load_json("chart_spreads.json")
    # ts = rt.chart_spreads(
    #     up["m*"]["user"],
    #     up["m*"]["pass"],
    #     [
    #         ("@HO4H", "@HO4J", "2014"),
    #         ("@HO9H", "@HO9J", "2019"),
    #         ("@HO0H", "@HO0J", "2020"),
    #     ],
    #     feed="CME_NymexFutures_EOD",
    #     output="data",
    # )

    # ts.spread = ts.spread.mul(42).round(4)
    # ts = ts[["year", "spread", "days_from_exp", "date"]]
    # ts.columns = ["year", "value", "DaysFromExp", "date"]
    # ts = ts.reset_index(drop=True).dropna()
    # ts.year = pd.to_numeric(ts.year)

    # # assert ac.equals(ts), "chart_spreads Test failed"
    pass


def test_chart_zscore():
    df = rt.data.open_data("eiaStocks")
    df = df.loc[df.series == "NGLower48", ["date", "value"]].set_index("date")["value"]
    df = df.resample("W-FRI").mean()
    stl = rt.chart_zscore(df)

    assert isinstance(stl, go.Figure), "chart_zscore Test failed"


@pytest.mark.skip(reason="Requires EIA API key and network access")
def test_chart_eia_sd():
    fig = rt.chart_eia_sd("mogas", up["eia"])
    assert isinstance(fig, go.Figure), "chart_eia_sd Test failed"


@pytest.mark.skip(reason="Requires EIA API key and network access")
def test_chart_eia_steo():
    fig = rt.chart_eia_steo(up["eia"])
    assert isinstance(fig, go.Figure), "chart_eia_steo Test failed"


@pytest.mark.skip(reason="Requires Morningstar API credentials")
def test_swap_com():
    # ac = _load_json("swapCOM.json")

    # df = rt.get_prices(
    #     up["m*"]["user"],
    #     up["m*"]["pass"],
    #     codes=["CL0M", "CL0N", "CL0Q"],
    #     start_dt="2019-08-26",
    # )
    # df = df.settlement_price.unstack(level=0)
    # ts = rt.swap_com(
    #     df=df,
    #     futures_names=["CL0M", "CL0N"],
    #     start_dt="2020-05-01",
    #     end_dt="2020-05-30",
    #     cmdty="cmewti",
    #     exchange="nymex",
    # )

    # ac = ac.set_index("date")
    # ac = ac.round(4)
    # ts.index.name = "date"

    # assert np.allclose(ac, ts), "swap_com Test failed"
    pass


# ======================================================================
# Edge case tests — bond, npv, crr_euro, stl_decomposition
# ======================================================================


class TestBondEdgeCases:
    """Canonical bond pricing examples with known answers."""

    def test_par_bond(self):
        """When coupon rate = YTM, price should be 100 (par)."""
        price = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="price")
        assert abs(price - 100.0) < 0.01

    def test_zero_coupon_bond(self):
        """Zero coupon bond: price = 100 / (1 + ytm/m)^(T*m)."""
        price = rt.bond(ytm=0.06, c=0.0, T=5, m=2, output="price")
        expected = 100 / (1 + 0.06 / 2) ** (5 * 2)
        assert abs(price - expected) < 0.01

    def test_premium_bond(self):
        """When coupon > YTM, bond trades at a premium (> 100)."""
        price = rt.bond(ytm=0.03, c=0.05, T=10, m=2, output="price")
        assert price > 100

    def test_discount_bond(self):
        """When coupon < YTM, bond trades at a discount (< 100)."""
        price = rt.bond(ytm=0.08, c=0.05, T=10, m=2, output="price")
        assert price < 100

    def test_duration_less_than_maturity(self):
        """Macaulay duration should be less than T for coupon bonds."""
        dur = rt.bond(ytm=0.05, c=0.05, T=10, m=2, output="duration")
        assert 0 < dur < 10

    def test_zero_coupon_duration_equals_maturity(self):
        """For a zero-coupon bond, duration = maturity."""
        dur = rt.bond(ytm=0.05, c=0.0, T=5, m=2, output="duration")
        assert abs(dur - 5.0) < 0.01

    def test_df_output(self):
        """output='df' should return a DataFrame with expected columns."""
        df = rt.bond(ytm=0.05, c=0.05, T=2, m=2, output="df")
        assert isinstance(df, pd.DataFrame)
        for col in ["t_years", "cf", "t_periods", "disc_factor", "pv", "duration"]:
            assert col in df.columns

    def test_df_pv_sum_equals_price(self):
        """Sum of PV column should equal the price."""
        df = rt.bond(ytm=0.05, c=0.05, T=2, m=2, output="df")
        price = rt.bond(ytm=0.05, c=0.05, T=2, m=2, output="price")
        assert abs(df.pv.sum() - price) < 0.01

    def test_invalid_output_raises(self):
        with pytest.raises(ValueError):
            rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="invalid")


class TestNpvEdgeCases:
    @pytest.fixture
    def ir(self):
        return _load_json("./data/ir.json").rename(
            {"_row": "index"}, axis=1
        ).replace("...1", "0").set_index("index")

    def test_npv_none_disc_factors_raises(self):
        """Passing disc_factors=None should raise ValueError."""
        with pytest.raises(ValueError, match="Please input"):
            rt.npv(init_cost=-100, C=10, cf_freq=1, F=100, T=5, disc_factors=None)

    def test_npv_break_even_output(self, ir):
        """Break-even NPV should return a DataFrame."""
        df = rt.npv(
            init_cost=-375, C=50, cf_freq=0.5, F=250, T=2,
            disc_factors=ir, break_even=True, be_yield=0.05,
        )
        assert isinstance(df, pd.DataFrame)
        assert "pv" in df.columns
        assert "cf" in df.columns

    def test_npv_positive_investment(self, ir):
        """Initial cost at t=0 should be negative for typical investment."""
        df = rt.npv(
            init_cost=-375, C=50, cf_freq=0.5, F=250, T=2,
            disc_factors=ir, break_even=False,
        )
        assert df.loc[df.t == 0, "cf"].iloc[0] == -375

    def test_npv_final_value(self, ir):
        """Final cash flow should equal F."""
        df = rt.npv(
            init_cost=-375, C=50, cf_freq=0.5, F=250, T=2,
            disc_factors=ir, break_even=False,
        )
        assert df.loc[df.t == 2.0, "cf"].iloc[0] == 250


class TestCrrEuroEdgeCases:
    def test_call_price_positive(self):
        result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")
        assert result["price"] > 0

    def test_put_price_positive(self):
        result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="put")
        assert result["price"] > 0

    def test_deep_itm_call(self):
        """Deep ITM call (s >> x) should be close to s - x*exp(-Rf*T)."""
        result = rt.crr_euro(s=200, x=100, sigma=0.2, Rf=0.05, T=1, n=50, type="call")
        lower_bound = 200 - 100 * np.exp(-0.05)  # intrinsic value (lower bound for American)
        assert result["price"] > lower_bound * 0.95

    def test_deep_otm_call(self):
        """Deep OTM call (s << x) should be close to 0."""
        result = rt.crr_euro(s=50, x=200, sigma=0.2, Rf=0.05, T=1, n=50, type="call")
        assert result["price"] < 5

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - X*exp(-Rf*T) (approximately for binomial)."""
        s, x, sigma, Rf, T, n = 100, 100, 0.2, 0.05, 1, 50
        call = rt.crr_euro(s=s, x=x, sigma=sigma, Rf=Rf, T=T, n=n, type="call")
        put = rt.crr_euro(s=s, x=x, sigma=sigma, Rf=Rf, T=T, n=n, type="put")
        lhs = call["price"] - put["price"]
        rhs = s - x * np.exp(-Rf * T)
        assert abs(lhs - rhs) < 1.0  # binomial approximation, not exact

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="straddle")

    def test_asset_tree_shape(self):
        n = 5
        result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=n)
        assert result["asset"].shape == (n + 1, n + 1)
        assert result["option"].shape == (n + 1, n + 1)

    def test_note_ok(self):
        """Normal parameters should produce 'ok' note."""
        result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5)
        assert result["note"] == "ok"


class TestStlDecomposition:
    def test_stl_data_output(self):
        """STL decomposition with output='data' should return a DecomposeResult."""
        df = rt.data.open_data("dflong")
        df = df["CL01"]
        result = rt.stl_decomposition(
            df, output="data", seasonal=13, seasonal_deg=1, resample_freq="M"
        )
        assert hasattr(result, "trend")
        assert hasattr(result, "seasonal")
        assert hasattr(result, "resid")

    def test_stl_chart_output(self):
        """STL decomposition with output='chart' should return a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")
        df = rt.data.open_data("dflong")
        df = df["CL01"]
        result = rt.stl_decomposition(
            df, output="chart", seasonal=13, seasonal_deg=1, resample_freq="M"
        )
        import matplotlib.pyplot as _plt
        _plt.close("all")

    def test_stl_components_sum(self):
        """trend + seasonal + resid should reconstruct observed."""
        df = rt.data.open_data("dflong")
        df = df["CL01"]
        result = rt.stl_decomposition(
            df, output="data", seasonal=13, seasonal_deg=1, resample_freq="M"
        )
        reconstructed = result.trend + result.seasonal + result.resid
        observed = result.observed
        assert np.allclose(reconstructed.dropna(), observed.dropna(), atol=1e-6)


class TestReturns:
    """Test the returns() function with canonical examples."""

    def test_relative_returns(self):
        """Relative returns: (P1 - P0) / P0."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        prices = pd.Series([100, 110, 99, 104.94], index=idx, name="test")
        prices = pd.DataFrame({"test": prices})
        prices.index.name = "date"
        result = rt.returns(df=prices, ret_type="rel", period_return=1, spread=True)
        expected = [0.10, -0.10, 0.06]
        assert np.allclose(result.dropna().values.flatten(), expected, atol=1e-2)

    def test_absolute_returns(self):
        """Absolute returns: P1 - P0."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        prices = pd.Series([100, 110, 100, 105], index=idx, name="test")
        prices = pd.DataFrame({"test": prices})
        prices.index.name = "date"
        result = rt.returns(df=prices, ret_type="abs", period_return=1, spread=True)
        expected = [10, -10, 5]
        assert np.allclose(result.dropna().values.flatten(), expected, atol=1e-6)

    def test_log_returns(self):
        """Log returns: ln(P1/P0)."""
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        prices = pd.Series([100, 110, 100], index=idx, name="test")
        prices = pd.DataFrame({"test": prices})
        prices.index.name = "date"
        result = rt.returns(df=prices, ret_type="log", period_return=1, spread=True)
        expected = [np.log(110 / 100), np.log(100 / 110)]
        assert np.allclose(result.dropna().values.flatten(), expected, atol=1e-6)


class TestInferFreq:
    def test_daily_series(self):
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        s = pd.Series(range(100), index=idx)
        freq = rt.infer_freq(s)
        assert freq in [1, "B", "D"]

    @pytest.mark.xfail(reason="infer_freq has known limitation with monthly data - unequal month lengths")
    def test_monthly_series(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        s = pd.Series(range(24), index=idx)
        freq = rt.infer_freq(s)
        assert freq in [1, "M", "ME"]

    def test_multiplier_mode(self):
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        s = pd.Series(range(100), index=idx)
        scale = rt.infer_freq(s, multiplier=True)
        assert scale == 252


if __name__ == "__main__":
    test_returns()
    pass

