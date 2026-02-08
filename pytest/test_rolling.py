"""Tests for _rolling.py — rolling analytics functions."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    return pd.Series(np.random.normal(0.0004, 0.01, len(idx)), index=idx)


@pytest.fixture
def benchmark_returns():
    np.random.seed(123)
    idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    return pd.Series(np.random.normal(0.0003, 0.01, len(idx)), index=idx)


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------
class TestRollingSharpe:
    def test_nan_count(self, daily_returns):
        sr = rt.rolling_sharpe(daily_returns, window=63)
        assert sr.isna().sum() == 62

    def test_annualized_vs_raw(self, daily_returns):
        sr_ann = rt.rolling_sharpe(daily_returns, window=63, annualize=True)
        sr_raw = rt.rolling_sharpe(daily_returns, window=63, annualize=False)
        ratio = sr_ann.dropna().iloc[-1] / sr_raw.dropna().iloc[-1]
        assert abs(ratio - np.sqrt(252)) < 0.01

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            rt.rolling_sharpe([1, 2, 3])


# ---------------------------------------------------------------------------
# rolling_beta
# ---------------------------------------------------------------------------
class TestRollingBeta:
    def test_nan_count(self, daily_returns, benchmark_returns):
        beta = rt.rolling_beta(daily_returns, benchmark_returns, window=63)
        assert beta.isna().sum() == 62

    def test_known_beta(self):
        """Asset = 1.5 * benchmark + noise → beta ≈ 1.5."""
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
        bench = pd.Series(np.random.normal(0, 0.01, 500), index=idx)
        asset = pd.Series(
            1.5 * bench.values + np.random.normal(0, 0.003, 500), index=idx
        )
        beta = rt.rolling_beta(asset, bench, window=252)
        last_beta = beta.dropna().iloc[-1]
        assert abs(last_beta - 1.5) < 0.2


# ---------------------------------------------------------------------------
# rolling_correlation
# ---------------------------------------------------------------------------
class TestRollingCorrelation:
    def test_range(self, daily_returns, benchmark_returns):
        corr = rt.rolling_correlation(daily_returns, benchmark_returns, window=63)
        valid = corr.dropna()
        assert (valid >= -1 - 1e-10).all()
        assert (valid <= 1 + 1e-10).all()


# ---------------------------------------------------------------------------
# rolling_volatility
# ---------------------------------------------------------------------------
class TestRollingVolatility:
    def test_nan_count(self, daily_returns):
        vol = rt.rolling_volatility(daily_returns, window=21)
        assert vol.isna().sum() == 20

    def test_positive(self, daily_returns):
        vol = rt.rolling_volatility(daily_returns, window=21)
        assert (vol.dropna() > 0).all()


# ---------------------------------------------------------------------------
# rolling_skewness and rolling_kurtosis
# ---------------------------------------------------------------------------
class TestRollingMoments:
    def test_skewness_exists(self, daily_returns):
        sk = rt.rolling_skewness(daily_returns, window=126)
        assert sk.dropna().shape[0] > 0

    def test_kurtosis_exists(self, daily_returns):
        kt = rt.rolling_kurtosis(daily_returns, window=126)
        assert kt.dropna().shape[0] > 0

    def test_normal_skewness_near_zero(self):
        """Large window of normal returns → skewness near 0."""
        np.random.seed(0)
        idx = pd.bdate_range("2020-01-01", periods=5000, freq="B")
        ret = pd.Series(np.random.normal(0, 0.01, 5000), index=idx)
        sk = rt.rolling_skewness(ret, window=5000)
        assert abs(sk.dropna().iloc[-1]) < 0.3


# ---------------------------------------------------------------------------
# rolling_drawdown
# ---------------------------------------------------------------------------
class TestRollingDrawdown:
    def test_all_negative_or_zero(self, daily_returns):
        dd = rt.rolling_drawdown(daily_returns, window=63)
        assert (dd.dropna() <= 0).all()

    def test_nan_count(self, daily_returns):
        dd = rt.rolling_drawdown(daily_returns, window=63)
        assert dd.isna().sum() == 62


# ---------------------------------------------------------------------------
# rolling_sortino
# ---------------------------------------------------------------------------
class TestRollingSortino:
    def test_exists(self, daily_returns):
        so = rt.rolling_sortino(daily_returns, window=63)
        assert so.dropna().shape[0] > 0
