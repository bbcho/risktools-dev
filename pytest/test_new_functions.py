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

# Github Actions CI Env Vars
up = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": ""}

up["eia"] = os.getenv("EIA")
up["quandl"] = os.getenv("QUANDL")
up["m*"]["pass"] = os.getenv("MS_PASS")
up["m*"]["user"] = os.getenv("MS_USER")


def test_get_curves():
    cl = _load_json("getCurveCL.json")
    cl.expirationDate = pd.to_datetime(cl.expirationDate)

    bg = _load_json("getCurveBG.json")
    bg.expirationDate = pd.to_datetime(bg.expirationDate)

    df_cl = rt.get_curves(
        up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["CL"]
    )

    df_bg = rt.get_curves(
        up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["BG"]
    )

    pd.testing.assert_frame_equal(
        cl, df_cl, check_like=True
    ), "get_curves test failed on CL"

    pd.testing.assert_frame_equal(
        bg, df_bg, check_like=True
    ), "get_curves test failed on BG"

    cl["root"] = "CL"
    bg["root"] = "BG"
    combo = cl.append(bg)

    df = rt.get_curves(
        up["m*"]["user"],
        up["m*"]["pass"],
        date="2021-12-20",
        contract_roots=["CL", "BG"],
    )

    pd.testing.assert_frame_equal(
        combo, df, check_like=True
    ), "get_curves test failed on combined ['CL','BG']"

