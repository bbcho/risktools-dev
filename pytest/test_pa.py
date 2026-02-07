"""
Tests for risktools._pa — Performance Analytics functions.

Uses hand-calculable canonical examples where possible so that expected
values can be verified with a pocket calculator.
"""

import os, sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import risktools as rt


# ======================================================================
# return_cumulative
# ======================================================================

class TestReturnCumulative:
    """Canonical: geometric cumulative of [0.10, -0.05, 0.03]
    = (1.10)(0.95)(1.03) - 1 = 1.07635 - 1 = 0.07635
    Arithmetic: 0.10 + (-0.05) + 0.03 = 0.08
    """

    @pytest.fixture
    def simple(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        return pd.Series([0.10, -0.05, 0.03], index=idx)

    def test_geometric(self, simple):
        result = rt.return_cumulative(simple, geometric=True)
        expected = (1.10) * (0.95) * (1.03) - 1
        assert abs(result - expected) < 1e-10

    def test_arithmetic(self, simple):
        result = rt.return_cumulative(simple, geometric=False)
        assert abs(result - 0.08) < 1e-10

    def test_single_return(self):
        idx = pd.bdate_range("2020-01-02", periods=1, freq="B")
        s = pd.Series([0.05], index=idx)
        assert abs(rt.return_cumulative(s) - 0.05) < 1e-10

    def test_all_zeros(self):
        idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        s = pd.Series([0.0] * 5, index=idx)
        assert abs(rt.return_cumulative(s)) < 1e-10

    def test_dataframe(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        df = pd.DataFrame({"A": [0.10, -0.05, 0.03], "B": [0.01, 0.02, 0.03]}, index=idx)
        result = rt.return_cumulative(df, geometric=True)
        assert isinstance(result, pd.Series)
        expected_A = (1.10) * (0.95) * (1.03) - 1
        expected_B = (1.01) * (1.02) * (1.03) - 1
        assert abs(result["A"] - expected_A) < 1e-10
        assert abs(result["B"] - expected_B) < 1e-10


# ======================================================================
# return_annualized
# ======================================================================

class TestReturnAnnualized:
    """Canonical monthly example: 12 months of 1% monthly return.
    Geometric annualized = (1.01)^12 - 1 ≈ 0.12682503
    Arithmetic annualized = 0.01 * 12 = 0.12
    """

    @pytest.fixture
    def monthly_1pct(self):
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        return pd.Series([0.01] * 12, index=idx)

    def test_geometric(self, monthly_1pct):
        result = rt.return_annualized(monthly_1pct, geometric=True)
        expected = (1.01) ** 12 - 1
        assert abs(result - expected) < 1e-6

    def test_arithmetic(self, monthly_1pct):
        result = rt.return_annualized(monthly_1pct, geometric=False)
        assert abs(result - 0.12) < 1e-6

    def test_explicit_scale(self, monthly_1pct):
        # override scale to 12 (monthly)
        result = rt.return_annualized(monthly_1pct, scale=12, geometric=False)
        assert abs(result - 0.12) < 1e-6

    def test_daily(self):
        idx = pd.bdate_range("2020-01-02", periods=252, freq="B")
        # 252 days of 0.0004 daily return
        s = pd.Series([0.0004] * 252, index=idx)
        result = rt.return_annualized(s, geometric=False)
        expected = 0.0004 * 252
        assert abs(result - expected) < 1e-6


# ======================================================================
# return_excess
# ======================================================================

class TestReturnExcess:
    def test_scalar_rf(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        R = pd.Series([0.05, 0.03, -0.02], index=idx)
        result = rt.return_excess(R, Rf=0.01)
        expected = pd.Series([0.04, 0.02, -0.03], index=idx)
        pd.testing.assert_series_equal(result, expected)

    def test_zero_rf(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        R = pd.Series([0.05, 0.03, -0.02], index=idx)
        result = rt.return_excess(R, Rf=0)
        pd.testing.assert_series_equal(result, R)

    def test_series_rf(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        R = pd.Series([0.05, 0.03, -0.02], index=idx)
        Rf = pd.Series([0.001, 0.001, 0.001], index=idx)
        result = rt.return_excess(R, Rf)
        expected = R - Rf
        pd.testing.assert_series_equal(result, expected)


# ======================================================================
# sd_annualized
# ======================================================================

class TestSdAnnualized:
    """Canonical: if daily std = σ, annualized std = σ * √252."""

    def test_daily(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=1000, freq="B")
        s = pd.Series(np.random.normal(0, 0.01, 1000), index=idx)
        result = rt.sd_annualized(s)
        expected = s.std() * np.sqrt(252)
        assert abs(result - expected) < 1e-10

    def test_monthly(self):
        np.random.seed(42)
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        s = pd.Series(np.random.normal(0, 0.03, 60), index=idx)
        result = rt.sd_annualized(s)
        expected = s.std() * np.sqrt(12)
        assert abs(result - expected) < 1e-10

    def test_explicit_scale(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        s = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        result = rt.sd_annualized(s, scale=252)
        expected = s.std() * np.sqrt(252)
        assert abs(result - expected) < 1e-10

    def test_dataframe(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        df = pd.DataFrame(
            {"A": np.random.normal(0, 0.01, 100), "B": np.random.normal(0, 0.02, 100)},
            index=idx,
        )
        result = rt.sd_annualized(df)
        assert isinstance(result, pd.Series)
        assert abs(result["A"] - df["A"].std() * np.sqrt(252)) < 1e-10
        assert abs(result["B"] - df["B"].std() * np.sqrt(252)) < 1e-10

    def test_raises_non_datetime(self):
        s = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError):
            rt.sd_annualized(s)


# ======================================================================
# sharpe_ratio_annualized
# ======================================================================

class TestSharpeRatioAnnualized:
    """Sharpe = return_annualized(R-Rf) / sd_annualized(R)"""

    def test_zero_rf(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=252, freq="B")
        s = pd.Series(np.random.normal(0.0004, 0.01, 252), index=idx)
        result = rt.sharpe_ratio_annualized(s, Rf=0)
        # Manually compute
        ann_ret = rt.return_annualized(s, geometric=True)
        ann_sd = rt.sd_annualized(s)
        expected = ann_ret / ann_sd
        assert abs(result - expected) < 1e-10

    def test_with_rf(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=252, freq="B")
        s = pd.Series(np.random.normal(0.0004, 0.01, 252), index=idx)
        Rf = 0.0001  # daily risk-free rate
        result = rt.sharpe_ratio_annualized(s, Rf=Rf)
        excess = rt.return_excess(s, Rf)
        ann_ret = rt.return_annualized(excess, geometric=True)
        ann_sd = rt.sd_annualized(s)
        expected = ann_ret / ann_sd
        assert abs(result - expected) < 1e-10


# ======================================================================
# upside_risk / downside_deviation
# ======================================================================

class TestUpsideDownside:
    """Canonical: R = [0.05, -0.03, 0.02, -0.01, 0.04], MAR = 0
    Upside returns: [0.05, 0.02, 0.04]
    UpsideVariance (full) = (0.05^2 + 0.02^2 + 0.04^2) / 5 = (0.0025 + 0.0004 + 0.0016) / 5 = 0.0009
    UpsideRisk (full) = sqrt(0.0009) = 0.03
    UpsidePotential (full) = (0.05 + 0.02 + 0.04) / 5 = 0.022

    Downside returns: [-0.03, -0.01]
    DownsideDeviation (full) = sqrt((0.03^2 + 0.01^2) / 5) = sqrt(0.001/5) = sqrt(0.0002) ≈ 0.01414
    """

    @pytest.fixture
    def returns(self):
        idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        return pd.Series([0.05, -0.03, 0.02, -0.01, 0.04], index=idx)

    def test_upside_risk(self, returns):
        result = rt.upside_risk(returns, MAR=0, stat="risk")
        expected = np.sqrt((0.05**2 + 0.02**2 + 0.04**2) / 5)
        assert abs(result - expected) < 1e-10

    def test_upside_variance(self, returns):
        result = rt.upside_risk(returns, MAR=0, stat="variance")
        expected = (0.05**2 + 0.02**2 + 0.04**2) / 5
        assert abs(result - expected) < 1e-10

    def test_upside_potential(self, returns):
        result = rt.upside_risk(returns, MAR=0, stat="potential")
        expected = (0.05 + 0.02 + 0.04) / 5
        assert abs(result - expected) < 1e-10

    def test_downside_deviation(self, returns):
        result = rt.downside_deviation(returns, MAR=0)
        expected = np.sqrt((0.03**2 + 0.01**2) / 5)
        assert abs(result - expected) < 1e-10

    def test_downside_potential(self, returns):
        result = rt.downside_deviation(returns, MAR=0, potential=True)
        expected = (0.03 + 0.01) / 5
        assert abs(result - expected) < 1e-10

    def test_subset_method(self, returns):
        """With method='subset', denominator = number of downside returns."""
        result = rt.downside_deviation(returns, MAR=0, method="subset")
        expected = np.sqrt((0.03**2 + 0.01**2) / 2)
        assert abs(result - expected) < 1e-10

    def test_upside_subset(self, returns):
        result = rt.upside_risk(returns, MAR=0, method="subset", stat="risk")
        expected = np.sqrt((0.05**2 + 0.02**2 + 0.04**2) / 3)
        assert abs(result - expected) < 1e-10

    def test_nonzero_mar(self):
        idx = pd.bdate_range("2020-01-02", periods=4, freq="B")
        R = pd.Series([0.05, 0.01, -0.02, 0.03], index=idx)
        MAR = 0.02
        # Upside: [0.05, 0.03] (above 0.02)
        # UpsidePotential = ((0.05-0.02) + (0.03-0.02)) / 4 = (0.03 + 0.01) / 4 = 0.01
        result = rt.upside_risk(R, MAR=MAR, stat="potential")
        expected = (0.03 + 0.01) / 4
        assert abs(result - expected) < 1e-10


# ======================================================================
# omega_sharpe_ratio
# ======================================================================

class TestOmegaSharpeRatio:
    """Omega-Sharpe = (UpsidePotential - DownsidePotential) / DownsidePotential"""

    def test_basic(self):
        idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        R = pd.Series([0.05, -0.03, 0.02, -0.01, 0.04], index=idx)
        result = rt.omega_sharpe_ratio(R, MAR=0)
        up = rt.upside_risk(R, MAR=0, stat="potential")
        dp = rt.downside_deviation(R, MAR=0, potential=True)
        expected = (up - dp) / dp
        assert abs(result - expected) < 1e-10

    def test_all_positive(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        R = pd.Series([0.01, 0.02, 0.03], index=idx)
        # All returns above MAR=0, so downside_potential = 0
        # This should give inf or a very large number
        result = rt.omega_sharpe_ratio(R, MAR=0)
        assert result == float("inf") or result > 1e10


# ======================================================================
# drawdowns
# ======================================================================

class TestDrawdowns:
    """Canonical: R = [0.10, 0.05, -0.20, 0.05]
    Geometric cumulative wealth: 1.10, 1.155, 0.924, 0.9702
    Running max:                 1.10, 1.155, 1.155, 1.155
    Drawdown:                     0,    0,   -0.2, -0.16...
    """

    @pytest.fixture
    def returns(self):
        idx = pd.bdate_range("2020-01-02", periods=4, freq="B")
        return pd.Series([0.10, 0.05, -0.20, 0.05], index=idx)

    def test_geometric_drawdown(self, returns):
        dd = rt.drawdowns(returns, geometric=True)
        # First two periods: new highs, dd = 0
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-10)
        # Third period: wealth = 1.10*1.05*0.80 = 0.924, peak = 1.155
        # drawdown = 0.924/1.155 - 1 = -0.2
        assert dd.iloc[2] == pytest.approx(-0.2, abs=1e-10)

    def test_arithmetic_drawdown(self, returns):
        dd = rt.drawdowns(returns, geometric=False)
        # Arithmetic: cumsum+1 = [1.10, 1.15, 0.95, 1.00]
        # cummax (clipped) = [1.10, 1.15, 1.15, 1.15]
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-10)
        assert dd.iloc[2] < 0  # drawdown should be negative

    def test_no_drawdown(self):
        idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        s = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02], index=idx)
        dd = rt.drawdowns(s)
        # all positive => all cumulative returns are new highs => dd = 0
        assert (dd >= -1e-10).all()

    def test_dataframe(self):
        idx = pd.bdate_range("2020-01-02", periods=3, freq="B")
        df = pd.DataFrame({"A": [0.10, -0.20, 0.05], "B": [0.05, 0.10, -0.20]}, index=idx)
        dd = rt.drawdowns(df)
        assert isinstance(dd, pd.DataFrame)
        assert dd.shape == (3, 2)


# ======================================================================
# find_drawdowns
# ======================================================================

class TestFindDrawdowns:
    def test_single_drawdown(self):
        idx = pd.bdate_range("2020-01-02", periods=6, freq="B")
        # Up, up, down, down, up, up  => one drawdown episode
        R = pd.Series([0.05, 0.05, -0.10, -0.05, 0.08, 0.08], index=idx)
        result = rt.find_drawdowns(R)
        # Should contain arrays for 'return', 'from', 'trough', 'to', etc.
        assert "return" in result
        assert "from" in result
        assert "trough" in result
        assert "length" in result
        # The min return should be negative
        assert result["return"].min() < 0

    def test_dataframe_input(self):
        idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [0.05, -0.10, -0.05, 0.08, 0.03], "B": [-0.05, 0.10, 0.05, -0.08, 0.03]},
            index=idx,
        )
        result = rt.find_drawdowns(df)
        assert "A" in result
        assert "B" in result


# ======================================================================
# CAPM_beta
# ======================================================================

class TestCAPMBeta:
    """Canonical: if Ra = 2*Rb, then beta = 2.
    Create Rb, and Ra = 2*Rb (no noise).
    """

    @pytest.fixture
    def perfect_beta2(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        Rb = pd.Series(np.random.normal(0, 0.01, 100), index=idx, name="benchmark")
        Ra = 2 * Rb
        Ra.name = "asset"
        return Ra, Rb

    def test_beta_all(self, perfect_beta2):
        Ra, Rb = perfect_beta2
        beta = rt.CAPM_beta(Ra, Rb, Rf=0, kind="all")
        assert abs(beta - 2.0) < 1e-6

    def test_beta_bull(self, perfect_beta2):
        Ra, Rb = perfect_beta2
        beta = rt.CAPM_beta(Ra, Rb, Rf=0, kind="bull")
        assert abs(beta - 2.0) < 1e-6

    def test_beta_bear(self, perfect_beta2):
        Ra, Rb = perfect_beta2
        beta = rt.CAPM_beta(Ra, Rb, Rf=0, kind="bear")
        assert abs(beta - 2.0) < 1e-6

    def test_beta_one_identity(self):
        """Beta of an asset against itself should be 1.0."""
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        R = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        beta = rt.CAPM_beta(R, R, Rf=0)
        assert abs(beta - 1.0) < 1e-6

    def test_dataframe_input(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=100, freq="B")
        Rb = pd.Series(np.random.normal(0, 0.01, 100), index=idx, name="bench")
        Ra = pd.DataFrame({"X": 1.5 * Rb, "Y": 0.5 * Rb}, index=idx)
        result = rt.CAPM_beta(Ra, Rb, Rf=0)
        assert isinstance(result, pd.Series)
        assert abs(result["X"] - 1.5) < 1e-5
        assert abs(result["Y"] - 0.5) < 1e-5


# ======================================================================
# timing_ratio
# ======================================================================

class TestTimingRatio:
    """TimingRatio = beta_bull / beta_bear. For Ra = 2*Rb, ratio = 1.0"""

    def test_symmetric_beta(self):
        np.random.seed(42)
        idx = pd.bdate_range("2020-01-02", periods=200, freq="B")
        Rb = pd.Series(np.random.normal(0, 0.01, 200), index=idx)
        Ra = 2 * Rb
        result = rt.timing_ratio(Ra, Rb, Rf=0)
        assert abs(result - 1.0) < 0.05  # allow small deviation from sample noise


# ======================================================================
# _resolve_scale
# ======================================================================

class TestResolveScale:
    def test_business_day(self):
        from risktools._pa import _resolve_scale
        assert _resolve_scale("B") == 252

    def test_monthly(self):
        from risktools._pa import _resolve_scale
        assert _resolve_scale("M") == 12

    def test_weekly_prefix(self):
        from risktools._pa import _resolve_scale
        assert _resolve_scale("W-FRI") == 52

    def test_unknown_raises(self):
        from risktools._pa import _resolve_scale
        with pytest.raises(ValueError):
            _resolve_scale("X")
