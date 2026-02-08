"""Tests for _spreads.py — commodity spread analytics."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=5, freq="B")


# ---------------------------------------------------------------------------
# crack_spread
# ---------------------------------------------------------------------------
class TestCrackSpread:
    def test_3_2_1(self, dates):
        crude = pd.Series([70, 71, 72, 73, 74.0], index=dates)
        gas = pd.Series([2.5, 2.55, 2.6, 2.65, 2.7], index=dates)
        ho = pd.Series([2.8, 2.85, 2.9, 2.95, 3.0], index=dates)
        cs = rt.crack_spread(crude, gas, ho, "3:2:1")
        # Manual: (2*2.5*42 + 1*2.8*42 - 3*70) / 3 = (210+117.6-210)/3 = 39.2
        assert abs(cs.iloc[0] - 39.2) < 0.001

    def test_1_1(self, dates):
        crude = pd.Series([70, 71, 72, 73, 74.0], index=dates)
        gas = pd.Series([2.5, 2.55, 2.6, 2.65, 2.7], index=dates)
        cs = rt.crack_spread(crude, gas, ratio="1:1")
        # 2.5*42 - 70 = 105 - 70 = 35
        assert abs(cs.iloc[0] - 35.0) < 0.001

    def test_5_3_2(self, dates):
        crude = pd.Series([70.0] * 5, index=dates)
        gas = pd.Series([2.5] * 5, index=dates)
        ho = pd.Series([2.8] * 5, index=dates)
        cs = rt.crack_spread(crude, gas, ho, "5:3:2")
        # (3*2.5*42 + 2*2.8*42 - 5*70) / 5 = (315+235.2-350)/5 = 40.04
        assert abs(cs.iloc[0] - 40.04) < 0.001

    def test_missing_ho_raises(self, dates):
        crude = pd.Series([70.0] * 5, index=dates)
        gas = pd.Series([2.5] * 5, index=dates)
        with pytest.raises(ValueError, match="heating_oil"):
            rt.crack_spread(crude, gas, ratio="3:2:1")

    def test_invalid_ratio(self, dates):
        crude = pd.Series([70.0] * 5, index=dates)
        gas = pd.Series([2.5] * 5, index=dates)
        with pytest.raises(ValueError, match="ratio"):
            rt.crack_spread(crude, gas, ratio="4:3:1")


# ---------------------------------------------------------------------------
# spark_spread
# ---------------------------------------------------------------------------
class TestSparkSpread:
    def test_basic(self, dates):
        power = pd.Series([45, 50, 48, 52, 46.0], index=dates)
        gas = pd.Series([3.5, 3.6, 3.55, 3.65, 3.5], index=dates)
        ss = rt.spark_spread(power, gas, heat_rate=7.0)
        # 45 - 3.5*7 = 20.5
        assert abs(ss.iloc[0] - 20.5) < 0.001

    def test_negative_heat_rate(self, dates):
        power = pd.Series([45.0] * 5, index=dates)
        gas = pd.Series([3.5] * 5, index=dates)
        with pytest.raises(ValueError, match="heat_rate"):
            rt.spark_spread(power, gas, heat_rate=-1)


# ---------------------------------------------------------------------------
# crush_spread
# ---------------------------------------------------------------------------
class TestCrushSpread:
    def test_basic(self, dates):
        beans = pd.Series([1300.0] * 5, index=dates)
        meal = pd.Series([380.0] * 5, index=dates)
        oil = pd.Series([55.0] * 5, index=dates)
        cs = rt.crush_spread(beans, meal, oil)
        # meal_value = 380 * 0.024 = 9.12
        # oil_value = 55 * 11 / 100 = 6.05
        # bean_cost = 1300 / 100 = 13.0
        # crush = 9.12 + 6.05 - 13.0 = 2.17
        assert abs(cs.iloc[0] - 2.17) < 0.001


# ---------------------------------------------------------------------------
# convenience_yield
# ---------------------------------------------------------------------------
class TestConvenienceYield:
    def test_known_value(self):
        # F=102, S=100, r=0.05, T=0.5
        # y = 0.05 - ln(102/100)/0.5
        y = rt.convenience_yield(100.0, 102.0, 0.05, 0.5)
        expected = 0.05 - np.log(102 / 100) / 0.5
        assert abs(y - expected) < 1e-10

    def test_backwardation_positive_yield(self):
        # F < S*(1+r)^T → positive convenience yield
        y = rt.convenience_yield(100.0, 98.0, 0.05, 1.0)
        assert y > 0

    def test_series_inputs(self, dates):
        spot = pd.Series([100, 101, 102, 103, 104.0], index=dates)
        fut = pd.Series([102, 103, 104, 105, 106.0], index=dates)
        y = rt.convenience_yield(spot, fut, 0.05, 0.5)
        assert len(y) == 5


# ---------------------------------------------------------------------------
# optimal_hedge_ratio
# ---------------------------------------------------------------------------
class TestOptimalHedgeRatio:
    def test_perfect_hedge(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        spot = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        fut = spot + pd.Series(np.random.normal(0, 0.002, 200), index=dates)
        r = rt.optimal_hedge_ratio(spot, fut)
        assert abs(r["hedge_ratio"] - 1.0) < 0.1
        assert r["r_squared"] > 0.9

    def test_ols_vs_min_variance(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        spot = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        fut = 0.8 * spot + pd.Series(np.random.normal(0, 0.01, 200), index=dates)
        r_ols = rt.optimal_hedge_ratio(spot, fut, method="ols")
        r_mv = rt.optimal_hedge_ratio(spot, fut, method="min_variance")
        assert abs(r_ols["hedge_ratio"] - r_mv["hedge_ratio"]) < 0.05


# ---------------------------------------------------------------------------
# calendar_spread and fly_spread
# ---------------------------------------------------------------------------
class TestCalendarAndFly:
    def test_calendar_spread(self, dates):
        front = pd.Series([75, 74.5, 76, 75.5, 77.0], index=dates)
        back = pd.Series([73, 73.5, 74, 74.5, 75.0], index=dates)
        cs = rt.calendar_spread(front, back)
        assert abs(cs.iloc[0] - 2.0) < 1e-10

    def test_fly_spread(self, dates):
        front = pd.Series([75, 74.5, 76, 75.5, 77.0], index=dates)
        middle = pd.Series([74, 73.5, 75, 74.5, 76.0], index=dates)
        back = pd.Series([73.5, 73.0, 74.5, 74.0, 75.5], index=dates)
        fly = rt.fly_spread(front, middle, back)
        # 75 - 2*74 + 73.5 = 0.5
        assert abs(fly.iloc[0] - 0.5) < 1e-10
