"""
Tests for risktools._cullenfrey — Cullen and Frey distribution analysis.

Uses canonical distributions with known skewness and kurtosis values.
"""

import os, sys
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import matplotlib.figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import risktools as rt


class TestDescribeDistribution:
    """
    Canonical examples:
    - Normal distribution: skewness ≈ 0, kurtosis ≈ 3
    - Uniform distribution: skewness = 0, kurtosis = 1.8
    - Exponential distribution: skewness = 2, kurtosis = 9
    """

    @pytest.fixture
    def normal_data(self):
        np.random.seed(42)
        return np.random.normal(0, 1, 10000)

    @pytest.fixture
    def uniform_data(self):
        np.random.seed(42)
        return np.random.uniform(0, 1, 10000)

    @pytest.fixture
    def exponential_data(self):
        np.random.seed(42)
        return np.random.exponential(1, 10000)

    def test_normal_stats(self, normal_data):
        """Normal: skewness ≈ 0, kurtosis ≈ 3."""
        res = rt.describe_distribution(normal_data, graph=False, method="unbiased")
        assert isinstance(res, dict)
        assert abs(res["skewness"]) < 0.1
        assert abs(res["kurtosis"] - 3.0) < 0.2

    def test_uniform_stats(self, uniform_data):
        """Uniform: skewness ≈ 0, kurtosis ≈ 1.8."""
        res = rt.describe_distribution(uniform_data, graph=False, method="unbiased")
        assert abs(res["skewness"]) < 0.1
        assert abs(res["kurtosis"] - 1.8) < 0.2

    def test_exponential_stats(self, exponential_data):
        """Exponential: skewness ≈ 2, kurtosis ≈ 9."""
        res = rt.describe_distribution(exponential_data, graph=False, method="unbiased")
        assert abs(res["skewness"] - 2.0) < 0.3
        assert abs(res["kurtosis"] - 9.0) < 1.5

    def test_sample_vs_unbiased(self, normal_data):
        """Sample and unbiased methods should give different but close results for large n."""
        res_ub = rt.describe_distribution(normal_data, graph=False, method="unbiased")
        res_s = rt.describe_distribution(normal_data, graph=False, method="sample")
        # Both should identify it as approximately normal
        assert abs(res_ub["skewness"]) < 0.1
        assert abs(res_s["skewness"]) < 0.1
        # Values should be close but not identical
        assert res_ub["sd"] != res_s["sd"]  # different formulas

    def test_stats_dict_keys(self, normal_data):
        res = rt.describe_distribution(normal_data, graph=False)
        expected_keys = {"min", "max", "median", "mean", "sd", "skewness", "kurtosis", "method"}
        assert set(res.keys()) == expected_keys

    def test_returns_figure(self, normal_data):
        """graph=True should return a matplotlib Figure."""
        result = rt.describe_distribution(normal_data, graph=True)
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close("all")

    def test_returns_axes_when_passed(self, normal_data):
        """If ax is provided, should return the Axes object."""
        fig, ax = plt.subplots()
        result = rt.describe_distribution(normal_data, graph=True, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close("all")

    def test_discrete_mode(self, normal_data):
        """discrete=True should produce a graph without error."""
        result = rt.describe_distribution(normal_data, graph=True, discrete=True)
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close("all")

    def test_bootstrap(self, normal_data):
        """boot parameter should enable bootstrapping without error."""
        result = rt.describe_distribution(normal_data, graph=True, boot=100)
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close("all")

    def test_small_sample(self):
        """Small sample data from docstring example."""
        x = [1, 4, 7, 9, 15, 20, 54]
        res = rt.describe_distribution(x, method="sample", graph=False)
        assert res["min"] == 1
        assert res["max"] == 54
        assert res["mean"] == pytest.approx(np.mean(x))

    def test_known_sample_stats(self):
        """Verify exact computation for a tiny, hand-calculable dataset.
        x = [1, 2, 3, 4, 5]
        mean = 3, sample std (pop) = sqrt(2) ≈ 1.4142
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = rt.describe_distribution(x, graph=False, method="sample")
        assert res["mean"] == pytest.approx(3.0)
        assert res["min"] == 1.0
        assert res["max"] == 5.0
        assert res["median"] == pytest.approx(3.0)
        # sample std (0 ddof) = sqrt(var) = sqrt(2)
        assert res["sd"] == pytest.approx(np.sqrt(2.0), abs=1e-6)
