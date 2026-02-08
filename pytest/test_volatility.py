"""Tests for _volatility.py — volatility estimators and analytics."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=504, freq="B")
    return pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)


@pytest.fixture
def ohlc_data():
    """Generate synthetic OHLC data."""
    np.random.seed(42)
    n = 504
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=idx)
    open_p = close.shift(1).bfill()
    high = pd.concat([open_p, close], axis=1).max(axis=1) + np.abs(
        np.random.normal(0, 0.5, n)
    )
    low = pd.concat([open_p, close], axis=1).min(axis=1) - np.abs(
        np.random.normal(0, 0.5, n)
    )
    return open_p, high, low, close


# ---------------------------------------------------------------------------
# realized_vol
# ---------------------------------------------------------------------------
class TestRealizedVol:
    def test_shape(self, daily_returns):
        rv = rt.realized_vol(daily_returns, window=21)
        assert len(rv) == len(daily_returns)
        assert rv.isna().sum() == 20

    def test_annualized_scale(self, daily_returns):
        rv_ann = rt.realized_vol(daily_returns, window=252, annualize=True)
        rv_raw = rt.realized_vol(daily_returns, window=252, annualize=False)
        ratio = rv_ann.dropna().iloc[-1] / rv_raw.dropna().iloc[-1]
        assert abs(ratio - np.sqrt(252)) < 0.01

    def test_known_vol(self):
        """Returns with known σ=0.01 daily → ~0.159 annualized."""
        np.random.seed(0)
        idx = pd.bdate_range("2020-01-01", periods=10000, freq="B")
        ret = pd.Series(np.random.normal(0, 0.01, 10000), index=idx)
        rv = rt.realized_vol(ret, window=5000)
        last = rv.dropna().iloc[-1]
        assert abs(last - 0.01 * np.sqrt(252)) < 0.005


# ---------------------------------------------------------------------------
# vol_cone
# ---------------------------------------------------------------------------
class TestVolCone:
    def test_shape(self, daily_returns):
        cone = rt.vol_cone(daily_returns)
        assert cone.shape == (5, 6)
        assert "current" in cone.columns
        assert "50%" in cone.columns

    def test_monotone_quantiles(self, daily_returns):
        cone = rt.vol_cone(daily_returns)
        # For each window row: 10% <= 25% <= 50% <= 75% <= 90%
        for _, row in cone.iterrows():
            assert row["10%"] <= row["25%"] + 1e-10
            assert row["25%"] <= row["50%"] + 1e-10
            assert row["50%"] <= row["75%"] + 1e-10
            assert row["75%"] <= row["90%"] + 1e-10

    def test_custom_windows(self, daily_returns):
        cone = rt.vol_cone(daily_returns, windows=[10, 21])
        assert cone.shape[0] == 2


# ---------------------------------------------------------------------------
# vol_term_structure
# ---------------------------------------------------------------------------
class TestVolTermStructure:
    def test_default_windows(self, daily_returns):
        ts = rt.vol_term_structure(daily_returns)
        assert ts.index.tolist() == [5, 10, 21, 42, 63, 126, 252]

    def test_all_positive(self, daily_returns):
        ts = rt.vol_term_structure(daily_returns)
        assert (ts > 0).all()


# ---------------------------------------------------------------------------
# parkinson_vol
# ---------------------------------------------------------------------------
class TestParkinsonVol:
    def test_shape(self, ohlc_data):
        _, h, l, _ = ohlc_data
        pv = rt.parkinson_vol(h, l, window=21)
        assert pv.isna().sum() == 20

    def test_positive(self, ohlc_data):
        _, h, l, _ = ohlc_data
        pv = rt.parkinson_vol(h, l, window=21)
        assert (pv.dropna() > 0).all()


# ---------------------------------------------------------------------------
# garman_klass_vol
# ---------------------------------------------------------------------------
class TestGarmanKlassVol:
    def test_has_values(self, ohlc_data):
        o, h, l, c = ohlc_data
        gk = rt.garman_klass_vol(o, h, l, c, window=21)
        # Should have non-NaN values (with variance floor fix)
        assert gk.dropna().shape[0] > 0

    def test_non_negative(self, ohlc_data):
        o, h, l, c = ohlc_data
        gk = rt.garman_klass_vol(o, h, l, c, window=21)
        assert (gk.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# yang_zhang_vol
# ---------------------------------------------------------------------------
class TestYangZhangVol:
    def test_has_values(self, ohlc_data):
        o, h, l, c = ohlc_data
        yz = rt.yang_zhang_vol(o, h, l, c, window=21)
        assert yz.dropna().shape[0] > 0
