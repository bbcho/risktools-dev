"""Tests for _regimes.py — regime detection and structural breaks."""

import numpy as np
import pandas as pd
import pytest

import risktools as rt


# ---------------------------------------------------------------------------
# detect_regimes
# ---------------------------------------------------------------------------
class TestDetectRegimes:
    def test_two_regime_separation(self):
        """Two clearly separated regimes should be detected."""
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0.001, 0.005, 300)
        high_vol = rng.normal(-0.001, 0.02, 200)
        data = pd.Series(
            np.concatenate([low_vol, high_vol]),
            index=pd.bdate_range("2020-01-01", periods=500),
        )
        result = rt.detect_regimes(data, n_regimes=2, seed=42)
        assert len(result["means"]) == 2
        assert result["means"][0] < result["means"][1]
        assert result["states"].shape == (500,)
        assert result["transition_matrix"].shape == (2, 2)

    def test_means_sorted_ascending(self):
        rng = np.random.default_rng(42)
        data = pd.Series(
            rng.normal(0, 0.01, 500),
            index=pd.bdate_range("2020-01-01", periods=500),
        )
        result = rt.detect_regimes(data, n_regimes=3, seed=42)
        assert np.all(np.diff(result["means"]) >= 0)

    def test_transition_matrix_rows_sum_one(self):
        rng = np.random.default_rng(42)
        data = pd.Series(
            rng.normal(0, 0.01, 300),
            index=pd.bdate_range("2020-01-01", periods=300),
        )
        result = rt.detect_regimes(data, n_regimes=2, seed=42)
        row_sums = result["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_variances_positive(self):
        rng = np.random.default_rng(42)
        data = pd.Series(
            rng.normal(0, 0.01, 300),
            index=pd.bdate_range("2020-01-01", periods=300),
        )
        result = rt.detect_regimes(data, n_regimes=2, seed=42)
        assert (result["variances"] > 0).all()

    def test_invalid_method(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        with pytest.raises(ValueError, match="Unknown method"):
            rt.detect_regimes(data, method="kmeans")

    def test_reproducibility(self):
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0, 0.01, 200))
        r1 = rt.detect_regimes(data, seed=42)
        r2 = rt.detect_regimes(data, seed=42)
        np.testing.assert_array_equal(r1["states"].values, r2["states"].values)


# ---------------------------------------------------------------------------
# structural_break
# ---------------------------------------------------------------------------
class TestStructuralBreak:
    def test_cusum_detects_mean_shift(self):
        rng = np.random.default_rng(0)
        vals = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
        s = pd.Series(vals, index=pd.date_range("2020-01-01", periods=200, freq="B"))
        result = rt.structural_break(s, method="cusum")
        assert len(result["breaks"]) > 0
        assert "cusum_pos" in result
        assert "cusum_neg" in result

    def test_constant_series_no_breaks(self):
        s = pd.Series(
            np.ones(100),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        result = rt.structural_break(s, method="cusum")
        assert len(result["breaks"]) == 0

    def test_cusum_sq(self):
        rng = np.random.default_rng(0)
        # Variance shift: low vol then high vol
        vals = np.concatenate([rng.normal(0, 0.5, 200), rng.normal(0, 3, 200)])
        s = pd.Series(vals, index=pd.date_range("2020-01-01", periods=400, freq="B"))
        result = rt.structural_break(s, method="cusum_sq")
        assert "breaks" in result
        assert "threshold" in result

    def test_invalid_method(self):
        s = pd.Series(np.ones(50))
        with pytest.raises(ValueError, match="Unknown method"):
            rt.structural_break(s, method="bayesian")


# ---------------------------------------------------------------------------
# regime_summary
# ---------------------------------------------------------------------------
class TestRegimeSummary:
    def test_output_columns(self):
        rng = np.random.default_rng(42)
        data = pd.Series(
            rng.normal(0.0005, 0.01, 500),
            index=pd.bdate_range("2020-01-01", periods=500),
        )
        result = rt.detect_regimes(data, n_regimes=2, seed=42)
        summary = rt.regime_summary(data, result["states"])
        expected_cols = {"mean", "volatility", "sharpe", "skewness", "kurtosis", "count", "pct"}
        assert set(summary.columns) == expected_cols

    def test_counts_sum_to_total(self):
        rng = np.random.default_rng(42)
        data = pd.Series(
            rng.normal(0, 0.01, 300),
            index=pd.bdate_range("2020-01-01", periods=300),
        )
        result = rt.detect_regimes(data, n_regimes=2, seed=42)
        summary = rt.regime_summary(data, result["states"])
        assert summary["count"].sum() == 300
        assert abs(summary["pct"].sum() - 100) < 1e-10
