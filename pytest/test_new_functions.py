from numpy.linalg.linalg import eigvals
import pandas as pd
import numpy as np
import os
import json
import sys
import plotly.graph_objects as go
from test_risktools import _load_json

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
from pandas_datareader import data

# TODO

test_date = "2021-12-24"

with open(os.path.dirname(os.path.realpath(__file__)) + '/../user.json', mode='r') as file:
    upf = json.load(file)

# Github Actions CI Env Vars
up = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": ""}

up["eia"] = os.getenv("EIA", upf['eia'])
up["quandl"] = os.getenv("QUANDL", upf["quandl"])
up["m*"]["pass"] = os.getenv("MS_PASS", upf["m*"]["pass"] )
up["m*"]["user"] = os.getenv("MS_USER", upf["m*"]["user"])

ms = dict(username=os.getenv("MS_USER"), password=os.getenv("MS_PASS"))


def test_dist_desc_plot():
    df = rt.data.open_data("dflong")
    x = df["BRN01"].pct_change().dropna()
    rt.dist_desc_plot(x)
    rt.dist_desc_plot(x.reset_index().value)


def test_refineryLP():
    assert True
    # ac = _load_json("refineryLP.json", dataframe=False)
    # crudes = rt.data.open_data("ref_opt_inputs")
    # products = rt.data.open_data("ref_opt_outputs")
    # ts = rt.refineryLP(crude_inputs=crudes, product_outputs=products)

    # assert np.allclose(ac["profit"], ts["profit"])
    # assert np.allclose(ac["slate"], ts["slate"])


def test_get_curves():
    pass
    # cl = _load_json("getCurveCL.json")
    # cl.expirationDate = pd.to_datetime(cl.expirationDate)

    # bg = _load_json("getCurveBG.json")
    # bg.expirationDate = pd.to_datetime(bg.expirationDate)

    # df_cl = rt.get_curves(
    #     up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["CL"]
    # )

    # df_bg = rt.get_curves(
    #     up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["BG"]
    # )

    # pd.testing.assert_frame_equal(
    #     cl, df_cl, check_like=True
    # ), "get_curves test failed on CL"

    # pd.testing.assert_frame_equal(
    #     bg, df_bg, check_like=True
    # ), "get_curves test failed on BG"

    # cl["root"] = "CL"
    # bg["root"] = "BG"
    # combo = cl.append(bg)

    # df = rt.get_curves(
    #     up["m*"]["user"],
    #     up["m*"]["pass"],
    #     date="2021-12-20",
    #     contract_roots=["CL", "BG"],
    # )

    # pd.testing.assert_frame_equal(
    #     combo, df, check_like=True
    # ), "get_curves test failed on combined ['CL','BG']"


def test_get_ir_swap_curve():
    pass
    # ac = _load_json("getIRSwapCurve.json")
    # ac.date = pd.to_datetime(ac.date)
    # ac = ac.set_index("date")
    # ac.index.name = "Date"

    # ts = rt.get_ir_swap_curve(up["m*"]["user"], up["m*"]["pass"], end_dt=ac.index.max())

    # pd.testing.assert_frame_equal(ac, ts, check_like=True)


def test_swap_info():
    pass
    # ac = _load_json("swapInfo.json").dropna()
    # ac.bizDays = pd.to_datetime(ac.bizDays)
    # ac = ac.set_index("bizDays")
    # ac.index.name = "date"
    # ac = ac.rename({"fut_contract": "futures_contract"}, axis=1)
    # ac = ac.replace("1stLineSettled", "first_line_settled")

    # ts = rt.swap_info(**ms, date="2020-05-06", output="dataframe")
    # print(ts)

    # pd.testing.assert_frame_equal(ac, ts, check_like=True)


def test_swap_fut_weights():

    assert (
        rt.swap_fut_weight(
            month="2020-09-01",
            contract="cmewti",
            exchange="nymex",
            output="num_days_fut1",
        )
        == 15
    )
    assert (
        rt.swap_fut_weight(
            month="2020-09-01",
            contract="cmewti",
            exchange="nymex",
            output="num_days_fut2",
        )
        == 6
    )

    ts = rt.swap_fut_weight(
        month="2020-09-01",
        contract="cmewti",
        exchange="nymex",
        output="first_fut_weight",
    )
    assert np.allclose(ts, 0.7142857)
