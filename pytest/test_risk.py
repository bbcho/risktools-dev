"""Tests for _risk.py — VaR, CVaR, CFaR, NPVaR, rolling risk measures."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import risktools as rt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_returns():
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=1000, freq="B")
    return pd.Series(np.random.normal(0.0004, 0.01, 1000), index=idx)


# ---------------------------------------------------------------------------
# value_at_risk
# ---------------------------------------------------------------------------
class TestValueAtRisk:
    def test_historical_is_negative(self, daily_returns):
        var = rt.value_at_risk(daily_returns, 0.95, "historical")
        assert var < 0

    def test_parametric_is_negative(self, daily_returns):
        var = rt.value_at_risk(daily_returns, 0.95, "parametric")
        assert var < 0

    def test_monte_carlo_is_negative(self, daily_returns):
        var = rt.value_at_risk(daily_returns, 0.95, "monte_carlo", random_state=42)
        assert var < 0

    def test_parametric_known_value(self):
        """For N(0,1) returns, VaR_95 should be approx -1.645."""
        np.random.seed(0)
        r = pd.Series(np.random.normal(0, 1, 100000))
        var = rt.value_at_risk(r, 0.95, "parametric")
        assert abs(var - norm.ppf(0.05)) < 0.05

    def test_higher_confidence_more_extreme(self, daily_returns):
        var_95 = rt.value_at_risk(daily_returns, 0.95, "historical")
        var_99 = rt.value_at_risk(daily_returns, 0.99, "historical")
        assert var_99 < var_95  # 99% VaR is more negative

    def test_invalid_method_raises(self, daily_returns):
        with pytest.raises(ValueError, match="method"):
            rt.value_at_risk(daily_returns, 0.95, "invalid")

    def test_invalid_confidence_raises(self, daily_returns):
        with pytest.raises(ValueError, match="confidence"):
            rt.value_at_risk(daily_returns, 1.5)


# ---------------------------------------------------------------------------
# cvar
# ---------------------------------------------------------------------------
class TestCVaR:
    def test_cvar_more_extreme_than_var(self, daily_returns):
        var = rt.value_at_risk(daily_returns, 0.95, "historical")
        es = rt.cvar(daily_returns, 0.95, "historical")
        assert es <= var

    def test_parametric_cvar_more_extreme(self, daily_returns):
        var = rt.value_at_risk(daily_returns, 0.95, "parametric")
        es = rt.cvar(daily_returns, 0.95, "parametric")
        assert es <= var

    def test_parametric_known_formula(self):
        """For N(0,1): CVaR_95 = -phi(z_0.05)/0.05 ≈ -2.0627."""
        np.random.seed(0)
        r = pd.Series(np.random.normal(0, 1, 50000))
        es = rt.cvar(r, 0.95, "parametric")
        expected = -norm.pdf(norm.ppf(0.05)) / 0.05
        assert abs(es - expected) < 0.1


# ---------------------------------------------------------------------------
# cfar
# ---------------------------------------------------------------------------
class TestCFaR:
    def test_simulated_paths(self):
        rng = np.random.default_rng(42)
        cf = pd.DataFrame(rng.normal(10, 3, (12, 1000)))
        result = rt.cfar(cf, confidence=0.95, horizon=12)
        assert result["cfar"] < result["expected_cf"]
        assert result["n_scenarios"] == 1000
        assert result["worst_case_cf"] <= result["cfar"]

    def test_historical_series(self):
        rng = np.random.default_rng(42)
        cf = pd.Series(
            rng.normal(10, 3, 120),
            index=pd.date_range("2015-01-01", periods=120, freq="ME"),
        )
        result = rt.cfar(cf, confidence=0.95, horizon=12)
        assert result["n_scenarios"] > 0
        assert result["cfar"] < result["expected_cf"]

    def test_horizon_none_uses_all(self):
        rng = np.random.default_rng(42)
        cf = pd.DataFrame(rng.normal(10, 3, (6, 500)))
        r1 = rt.cfar(cf, horizon=None)
        r2 = rt.cfar(cf, horizon=6)
        assert abs(r1["cfar"] - r2["cfar"]) < 1e-10

    def test_parametric_method(self):
        rng = np.random.default_rng(42)
        cf = pd.DataFrame(rng.normal(10, 3, (12, 1000)))
        result = rt.cfar(cf, method="parametric")
        assert result["cfar"] < result["expected_cf"]

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            rt.cfar([1, 2, 3])


# ---------------------------------------------------------------------------
# npv_at_risk
# ---------------------------------------------------------------------------
class TestNPVaR:
    def test_basic_npvar(self):
        rng = np.random.default_rng(42)
        cf = np.vstack([np.full(1000, -100.0), rng.normal(12, 3, (12, 1000))])
        cf_df = pd.DataFrame(cf)
        result = rt.npv_at_risk(cf_df, discount_rate=0.01, confidence=0.95)
        assert result["npv_at_risk"] < result["expected_npv"]
        assert result["npv_cvar"] <= result["npv_at_risk"]
        assert len(result["npv_distribution"]) == 1000
        assert "50%" in result["percentiles"]

    def test_constant_cashflows(self):
        """Constant CFs with constant rate → all NPVs identical."""
        cf = pd.DataFrame(np.full((5, 100), 10.0))
        result = rt.npv_at_risk(cf, discount_rate=0.05, confidence=0.95)
        assert result["std_npv"] < 1e-10
        assert abs(result["npv_at_risk"] - result["expected_npv"]) < 1e-10

    def test_series_input(self):
        cf = pd.Series([-100, 30, 30, 30, 30])
        result = rt.npv_at_risk(cf, discount_rate=0.10, confidence=0.95)
        # Single scenario, so VaR = CVaR = expected = actual NPV
        expected_npv = sum(
            cf.iloc[t] / (1.10 ** t) for t in range(5)
        )
        assert abs(result["expected_npv"] - expected_npv) < 0.01

    def test_stochastic_discount_rates(self):
        rng = np.random.default_rng(42)
        cf = pd.DataFrame(rng.normal(10, 1, (6, 500)))
        rates = pd.DataFrame(rng.uniform(0.01, 0.05, (6, 500)))
        result = rt.npv_at_risk(cf, discount_rate=rates)
        assert result["std_npv"] > 0

    def test_series_discount_rates(self):
        cf = pd.DataFrame(np.full((4, 200), 10.0))
        rates = pd.Series([0.05, 0.06, 0.07, 0.08])
        result = rt.npv_at_risk(cf, discount_rate=rates)
        assert result["std_npv"] < 1e-10  # all scenarios same


# ---------------------------------------------------------------------------
# rolling_var and rolling_cvar
# ---------------------------------------------------------------------------
class TestRollingRisk:
    def test_rolling_var_shape(self, daily_returns):
        rv = rt.rolling_var(daily_returns, window=63)
        assert len(rv) == len(daily_returns)
        assert rv.isna().sum() == 62

    def test_rolling_var_all_negative(self, daily_returns):
        rv = rt.rolling_var(daily_returns, window=63)
        assert (rv.dropna() < 0).all()

    def test_rolling_cvar_more_extreme(self, daily_returns):
        rv = rt.rolling_var(daily_returns, window=63, confidence=0.95)
        rc = rt.rolling_cvar(daily_returns, window=63, confidence=0.95)
        # CVaR should be <= VaR (more negative) at each point
        valid = rv.dropna().index
        assert (rc.loc[valid] <= rv.loc[valid] + 1e-12).all()

    def test_parametric_rolling_var(self, daily_returns):
        rv = rt.rolling_var(daily_returns, window=63, method="parametric")
        assert (rv.dropna() < 0).all()

    def test_parametric_rolling_cvar(self, daily_returns):
        rc = rt.rolling_cvar(daily_returns, window=63, method="parametric")
        assert rc.dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# parametric_var_portfolio
# ---------------------------------------------------------------------------
class TestParametricVaRPortfolio:
    def test_basic(self):
        w = np.array([0.6, 0.4])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0004]])
        var = rt.parametric_var_portfolio(w, cov, confidence=0.95)
        assert var < 0

    def test_with_portfolio_value(self):
        w = np.array([0.6, 0.4])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0004]])
        v1 = rt.parametric_var_portfolio(w, cov, portfolio_value=1.0)
        v2 = rt.parametric_var_portfolio(w, cov, portfolio_value=1e6)
        assert abs(v2 / v1 - 1e6) < 1

    def test_single_asset(self):
        w = np.array([1.0])
        cov = np.array([[0.04]])
        var = rt.parametric_var_portfolio(w, cov, confidence=0.95)
        expected = norm.ppf(0.05) * 0.2  # sigma = 0.2
        assert abs(var - expected) < 1e-10

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            rt.parametric_var_portfolio(
                np.array([0.5, 0.5]),
                np.array([[0.01]]),
            )
