"""Tests for _portfolio.py — portfolio optimization and risk."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


@pytest.fixture
def returns_3asset():
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
    return pd.DataFrame(
        np.random.randn(500, 3) * 0.01 + np.array([0.0003, 0.0005, 0.0002]),
        index=dates,
        columns=["A", "B", "C"],
    )


# ---------------------------------------------------------------------------
# portfolio_optimize
# ---------------------------------------------------------------------------
class TestPortfolioOptimize:
    def test_min_variance_weights_sum_one(self, returns_3asset):
        r = rt.portfolio_optimize(returns_3asset, method="min_variance")
        assert abs(r["weights"].sum() - 1.0) < 1e-6

    def test_min_variance_long_only(self, returns_3asset):
        r = rt.portfolio_optimize(returns_3asset, method="min_variance")
        assert (r["weights"] >= -1e-8).all()

    def test_max_sharpe_beats_min_var(self, returns_3asset):
        mv = rt.portfolio_optimize(returns_3asset, method="min_variance")
        ms = rt.portfolio_optimize(returns_3asset, method="max_sharpe")
        assert ms["sharpe_ratio"] >= mv["sharpe_ratio"] - 0.01  # allow tolerance

    def test_risk_parity_weights_sum_one(self, returns_3asset):
        r = rt.portfolio_optimize(returns_3asset, method="risk_parity")
        assert abs(r["weights"].sum() - 1.0) < 1e-6

    def test_target_return(self, returns_3asset):
        r = rt.portfolio_optimize(
            returns_3asset, method="target_return", target_return=0.10
        )
        # The achieved return should be close to the target
        assert abs(r["expected_return"] - 0.10) < 0.02

    def test_target_return_requires_target(self, returns_3asset):
        with pytest.raises(ValueError, match="target_return"):
            rt.portfolio_optimize(returns_3asset, method="target_return")

    def test_invalid_method(self, returns_3asset):
        with pytest.raises(ValueError, match="method"):
            rt.portfolio_optimize(returns_3asset, method="invalid")

    def test_return_keys(self, returns_3asset):
        r = rt.portfolio_optimize(returns_3asset)
        assert set(r.keys()) == {
            "weights", "expected_return", "volatility", "sharpe_ratio"
        }

    def test_single_asset(self):
        np.random.seed(0)
        dates = pd.bdate_range("2020-01-01", periods=100, freq="B")
        ret = pd.DataFrame(np.random.normal(0.001, 0.01, (100, 1)),
                           index=dates, columns=["X"])
        r = rt.portfolio_optimize(ret)
        assert abs(r["weights"].iloc[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# portfolio_var
# ---------------------------------------------------------------------------
class TestPortfolioVaR:
    def test_parametric_positive(self, returns_3asset):
        var = rt.portfolio_var(returns_3asset, [1/3, 1/3, 1/3])
        assert var > 0  # VaR expressed as positive loss

    def test_historical(self, returns_3asset):
        var = rt.portfolio_var(returns_3asset, [1/3, 1/3, 1/3], method="historical")
        assert var > 0

    def test_monte_carlo(self, returns_3asset):
        var = rt.portfolio_var(returns_3asset, [1/3, 1/3, 1/3], method="monte_carlo")
        assert var > 0

    def test_weight_mismatch_raises(self, returns_3asset):
        with pytest.raises(ValueError, match="weights"):
            rt.portfolio_var(returns_3asset, [0.5, 0.5])


# ---------------------------------------------------------------------------
# portfolio_performance
# ---------------------------------------------------------------------------
class TestPortfolioPerformance:
    def test_keys(self, returns_3asset):
        perf = rt.portfolio_performance(returns_3asset, [0.5, 0.3, 0.2])
        expected_keys = {
            "annualized_return", "annualized_volatility", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "sortino_ratio",
        }
        assert set(perf.keys()) == expected_keys

    def test_max_drawdown_positive(self, returns_3asset):
        perf = rt.portfolio_performance(returns_3asset, [0.5, 0.3, 0.2])
        assert perf["max_drawdown"] > 0

    def test_volatility_positive(self, returns_3asset):
        perf = rt.portfolio_performance(returns_3asset, [0.5, 0.3, 0.2])
        assert perf["annualized_volatility"] > 0
