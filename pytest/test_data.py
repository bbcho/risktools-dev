from numpy.linalg.linalg import eigvals
import pandas as pd
import numpy as np
import os
import json
import sys
import plotly.graph_objects as go
import time
import yfinance as yf

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
from pandas_datareader import data

# Test data

def test_crudeOil():

    dc = rt.data.open_data('crudeOil')

    assert dc['crudes'].shape[0] > 1, "crudeOil-crudes Data not loaded"

    assert dc['CanadianAssays'].shape[0] > 1, "crudeOil-CanadianAssays Data not loaded"

    assert dc['bpAssays'].shape[0] > 1, "crudeOil-bpAssays Data not loaded"

    assert dc['xomAssays'].shape[0] > 1, "crudeOil-xomAssays Data not loaded"

    assert dc['CanadaPrices'].shape[0] > 1, "crudeOil-CanadaPrices Data not loaded"

def test_dfwide_dflong():

    df = rt.data.open_data('dfwide')

    assert df.shape[0] > 1, "dfwide Data not loaded"

    df = rt.data.open_data('dflong')

    assert df.shape[0] > 1, "dflong Data not loaded"

def test_eiaStocks():

    df = rt.data.open_data('eiaStocks')

    assert df.shape[0] > 1, "eiaStocks Data not loaded"

def test_eiaStorageCap():

    df = rt.data.open_data('eiaStorageCap')

    assert df.shape[0] > 1, "eiaStorageCap Data not loaded"

def test_cushing():

    dc = rt.data.open_data('cushing')

    assert dc['c1'].shape[0] > 1, "cushing-c1 Data not loaded"

    assert dc['c2'].shape[0] > 1, "cushing-c2 Data not loaded"

    assert dc['c1c2'].shape[0] > 1, "cushing-c1c2 Data not loaded"

    assert dc['storage'].shape[0] > 1, "cushing-storage Data not loaded"

def test_eurodollar():

    df = rt.data.open_data('eurodollar')

    assert df.shape[0] > 1, "eurodollar Data not loaded"


def test_expiry_table():

    df = rt.data.open_data('expiry_table')

    assert df.shape[0] > 1, "expiry_table Data not loaded"


def test_fizdiffs():

    df = rt.data.open_data('fizdiffs')

    assert df.shape[0] > 1, "fizdiffs Data not loaded"


def test_futuresRef():

    dc = rt.data.open_data('futuresRef')

    assert dc['ContractMonths'].shape[0] > 1, "futuresRef-ContractMonths Data not loaded"

    assert dc['Specifications'].shape[0] > 1, "futuresRef-Specifications Data not loaded"


def test_fxfwd():

    dc = rt.data.open_data('fxfwd')

    assert dc['historical'].shape[0] > 1, "fxfwd-historical Data not loaded"

    assert dc['curve'].shape[0] > 1, "fxfwd-curve Data not loaded"


def test_holidaysOil():

    df = rt.data.open_data('holidaysOil')

    assert df.shape[0] > 1, "holidaysOil Data not loaded"


def test_ohlc():

    df = rt.data.open_data('ohlc')

    assert df.shape[0] > 1, "ohlc Data not loaded"


def test_planets():

    df = rt.data.open_data('planets')

    assert df.shape[0] > 1, "planets Data not loaded"


def test_refineryLPdata():

    df = rt.data.open_data('refineryLPdata')

    assert df['inputs'].shape[0] > 1, "refineryLPdata-inputs Data not loaded"

    assert df['outputs'].shape[0] > 1, "refineryLPdata-outputs Data not loaded"


def test_tickers_eia():

    df = rt.data.open_data('tickers_eia')

    assert df.shape[0] > 1, "tickers_eia Data not loaded"


def test_tradeCycle():

    df = rt.data.open_data('tradeCycle')

    assert df.shape[0] > 1, "tradeCycle Data not loaded"


def test_tradeHubs():

    df = rt.data.open_data('tradeHubs')

    assert df.shape[0] > 1, "tradeHubs Data not loaded"


def test_tradeprocess():

    df = rt.data.open_data('tradeprocess')

    assert df.shape[0] > 1, "tradeprocess Data not loaded"


def test_wti_swap():

    df = rt.data.open_data('wti_swap')

    assert df.shape[0] > 1, "wti_swap Data not loaded"


def test_stocks():

    df = rt.data.open_data('stocks')

    assert df['spy'].shape[0] > 1, "stocks-spy Data not loaded"

    assert df['uso'].shape[0] > 1, "stocks-uso Data not loaded"

    assert df['ry'].shape[0] > 1, "stocks-ry Data not loaded"


def test_tsQuotes():

    df = rt.data.open_data('tsQuotes')

    assert df.shape[0] >= 1, "tsQuotes-spy Data not loaded"


def test_usSwapCurves():

    df = rt.data.open_data('usSwapCurves')

    assert df['times'].shape[0] > 1, "usSwapCurves-times Data not loaded"

    assert df['discounts'].shape[0] > 1, "usSwapCurves-discounts Data not loaded"

    assert df['forwards'].shape[0] > 1, "usSwapCurves-forwards Data not loaded"

    assert df['zerorates'].shape[0] > 1, "usSwapCurves-zerorates Data not loaded"

    assert len(df['flatQuotes']) == 1, "usSwapCurves-flatQuotes Data not loaded"

    assert isinstance(df['params'], dict), "usSwapCurves-params Data not loaded"

    assert df['table'].shape[0] > 1, "usSwapCurves-table Data not loaded"


def test_usSwapCurvesPar():

    df = rt.data.open_data('usSwapCurvesPar')

    assert df['times'].shape[0] > 1, "usSwapCurvesPar-times Data not loaded"

    assert df['discounts'].shape[0] > 1, "usSwapCurvesPar-discounts Data not loaded"

    assert df['forwards'].shape[0] > 1, "usSwapCurvesPar-forwards Data not loaded"

    assert df['zerorates'].shape[0] > 1, "usSwapCurvesPar-zerorates Data not loaded"

    assert len(df['flatQuotes']) == 1, "usSwapCurvesPar-flatQuotes Data not loaded"

    assert isinstance(df['params'], dict), "usSwapCurvesPar-params Data not loaded"

    assert df['table'].shape[0] > 1, "usSwapCurvesPar-table Data not loaded"