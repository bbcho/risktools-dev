"""Tests for _curves.py — forward curve analytics."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=5, freq="B")


@pytest.fixture
def curve_df(dates):
    return pd.DataFrame(
        {"M1": [75, 74.5, 76, 75.5, 77],
         "M2": [73, 73.5, 74, 74.5, 75],
         "M3": [72, 72.5, 73, 73.5, 74]},
        index=dates,
    )


# ---------------------------------------------------------------------------
# curve_calendar_spread
# ---------------------------------------------------------------------------
class TestCurveCalendarSpread:
    def test_dataframe_input(self, curve_df):
        cs = rt.curve_calendar_spread(curve_df, front_idx=0, back_idx=1)
        assert abs(cs.iloc[0] - 2.0) < 1e-10  # 75 - 73

    def test_series_input(self):
        delivery = pd.date_range("2024-06-01", periods=6, freq="ME")
        curve = pd.Series([70, 71, 72, 73, 74, 75.0], index=delivery)
        cs = rt.curve_calendar_spread(curve)
        # First difference: 71-70=1, 72-71=1, etc.
        assert len(cs) == 5


# ---------------------------------------------------------------------------
# curve_fly
# ---------------------------------------------------------------------------
class TestCurveFly:
    def test_known_value(self, curve_df):
        fly = rt.curve_fly(curve_df, 0, 1, 2)
        # 75 - 2*73 + 72 = 1.0
        assert abs(fly.iloc[0] - 1.0) < 1e-10

    def test_flat_curve_zero_fly(self, dates):
        flat = pd.DataFrame(
            {"M1": [100.0] * 5, "M2": [100.0] * 5, "M3": [100.0] * 5},
            index=dates,
        )
        fly = rt.curve_fly(flat, 0, 1, 2)
        assert (abs(fly) < 1e-10).all()


# ---------------------------------------------------------------------------
# curve_shape
# ---------------------------------------------------------------------------
class TestCurveShape:
    def test_contango(self):
        delivery = pd.date_range("2024-06-01", periods=6, freq="ME")
        curve = pd.Series([70, 71, 72, 73, 74, 75.0], index=delivery)
        result = rt.curve_shape(curve)
        assert result["shape"] == "contango"
        assert result["slope"] > 0

    def test_backwardation(self):
        delivery = pd.date_range("2024-06-01", periods=6, freq="ME")
        curve = pd.Series([75, 74, 73, 72, 71, 70.0], index=delivery)
        result = rt.curve_shape(curve)
        assert result["shape"] == "backwardation"
        assert result["slope"] < 0

    def test_flat(self):
        delivery = pd.date_range("2024-06-01", periods=6, freq="ME")
        curve = pd.Series([70, 70, 70, 70, 70, 70.0], index=delivery)
        result = rt.curve_shape(curve)
        assert result["shape"] == "flat"

    def test_dataframe_input(self, dates):
        df = pd.DataFrame(
            {"M1": [70, 71, 72, 73, 74],
             "M2": [71, 72, 73, 74, 75],
             "M3": [72, 73, 74, 75, 76]},
            index=dates,
        )
        result = rt.curve_shape(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "shape" in result.columns


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------
class TestBasis:
    def test_basic(self, dates):
        spot = pd.Series([75, 74.5, 76, 75.5, 77.0], index=dates)
        fut = pd.Series([73, 73.5, 74, 74.5, 75.0], index=dates)
        b = rt.basis(spot, fut)
        assert abs(b.iloc[0] - 2.0) < 1e-10

    def test_backwardation_positive(self, dates):
        spot = pd.Series([80.0] * 5, index=dates)
        fut = pd.Series([75.0] * 5, index=dates)
        b = rt.basis(spot, fut)
        assert (b > 0).all()


# ---------------------------------------------------------------------------
# roll_yield
# ---------------------------------------------------------------------------
class TestRollYield:
    def test_known_value(self, dates):
        front = pd.Series([75.0] * 5, index=dates)
        next_m = pd.Series([73.0] * 5, index=dates)
        ry = rt.roll_yield(front, next_m)
        expected = (75 - 73) / 73
        assert abs(ry.iloc[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# term_structure_slope
# ---------------------------------------------------------------------------
class TestTermStructureSlope:
    def test_linear_contango(self):
        delivery = pd.date_range("2024-06-01", periods=12, freq="ME")
        curve = pd.Series(np.linspace(70, 82, 12), index=delivery)
        result = rt.term_structure_slope(curve, method="linear")
        assert result["slope"] > 0
        assert result["r_squared"] > 0.99

    def test_log_method(self):
        delivery = pd.date_range("2024-06-01", periods=12, freq="ME")
        curve = pd.Series(np.linspace(70, 82, 12), index=delivery)
        result = rt.term_structure_slope(curve, method="log")
        assert result["slope"] > 0


# ---------------------------------------------------------------------------
# curve_seasonality
# ---------------------------------------------------------------------------
class TestCurveSeasonality:
    def test_monthly(self):
        np.random.seed(42)
        idx = pd.date_range("2015-01-01", periods=120, freq="ME")
        seasonal = np.sin(np.arange(120) * 2 * np.pi / 12) * 5 + 50
        prices = pd.Series(seasonal + np.random.normal(0, 1, 120), index=idx)
        s = rt.curve_seasonality(prices, freq="M")
        assert s.shape == (12, 4)
        assert set(s.columns) == {"mean", "std", "count", "seasonal_factor"}
        # Each month should have 10 observations
        assert (s["count"] == 10).all()

    def test_quarterly(self):
        np.random.seed(42)
        idx = pd.date_range("2015-01-01", periods=40, freq="QE")
        prices = pd.Series(np.random.normal(50, 5, 40), index=idx)
        s = rt.curve_seasonality(prices, freq="Q")
        assert s.shape == (4, 4)
