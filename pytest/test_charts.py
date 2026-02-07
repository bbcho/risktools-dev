"""
Tests for risktools._charts — Charting functions.

Tests chart_zscore, chart_five_year_plot, chart_perf_summary,
chart_forward_curves, chart_pairs, and dist_desc_plot.
Functions requiring API credentials (chart_eia_sd, chart_eia_steo,
chart_spreads) are skipped.
"""

import os, sys
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import risktools as rt


# ======================================================================
# chart_zscore
# ======================================================================

class TestChartZscore:
    @pytest.fixture
    def weekly_series(self):
        """Weekly NG storage data for seasonal analysis."""
        df = rt.data.open_data("eiaStocks")
        df = df.loc[df.series == "NGLower48", ["date", "value"]].set_index("date")["value"]
        df = df.resample("W-FRI").mean()
        return df

    def test_returns_plotly_figure_zscore(self, weekly_series):
        fig = rt.chart_zscore(weekly_series, output="zscore")
        assert isinstance(fig, go.Figure)

    def test_returns_plotly_figure_seasonal(self, weekly_series):
        fig = rt.chart_zscore(weekly_series, output="seasonal")
        assert isinstance(fig, go.Figure)

    def test_returns_stl_object(self, weekly_series):
        stl = rt.chart_zscore(weekly_series, output="stl")
        # STL result should have .resid, .trend, .seasonal attributes
        assert hasattr(stl, "resid")
        assert hasattr(stl, "trend")
        assert hasattr(stl, "seasonal")

    def test_with_freq_resample(self, weekly_series):
        """Resampling to monthly should still work."""
        fig = rt.chart_zscore(weekly_series, freq="M", output="zscore")
        assert isinstance(fig, go.Figure)

    def test_stl_components_sum(self, weekly_series):
        """trend + seasonal + resid should reconstruct original (approximately)."""
        stl = rt.chart_zscore(weekly_series, output="stl")
        reconstructed = stl.trend + stl.seasonal + stl.resid
        original = stl.observed
        assert np.allclose(reconstructed.dropna(), original.dropna(), atol=1e-6)


# ======================================================================
# chart_five_year_plot
# ======================================================================

class TestChartFiveYearPlot:
    @pytest.fixture
    def cl01_series(self):
        df = rt.data.open_data("dfwide")
        return df["CL01"].dropna()

    def test_returns_plotly_figure(self, cl01_series):
        fig = rt.chart_five_year_plot(cl01_series)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, cl01_series):
        fig = rt.chart_five_year_plot(cl01_series)
        # Should have at least: min, max, current year, previous year, 5yr mean
        assert len(fig.data) >= 5

    def test_with_title(self, cl01_series):
        fig = rt.chart_five_year_plot(cl01_series, title="CL01 Seasonal")
        assert fig.layout.title.text == "CL01 Seasonal"


# ======================================================================
# chart_perf_summary
# ======================================================================

class TestChartPerfSummary:
    @pytest.fixture
    def return_data(self):
        df = rt.data.open_data("dfwide")
        df = df[["CL01", "CL12"]].dropna()
        return rt.returns(df, period_return=1)

    def test_returns_plotly_figure(self, return_data):
        fig = rt.chart_perf_summary(return_data)
        assert isinstance(fig, go.Figure)

    def test_geometric_and_arithmetic(self, return_data):
        fig_geo = rt.chart_perf_summary(return_data, geometric=True)
        fig_arith = rt.chart_perf_summary(return_data, geometric=False)
        assert isinstance(fig_geo, go.Figure)
        assert isinstance(fig_arith, go.Figure)

    def test_with_title(self, return_data):
        fig = rt.chart_perf_summary(return_data, title="Test Title")
        assert fig.layout.title.text == "Test Title"

    def test_has_two_subplots(self, return_data):
        """Should have cumulative returns (top) and drawdowns (bottom)."""
        fig = rt.chart_perf_summary(return_data)
        # Each asset produces 2 traces (one in returns subplot, one in drawdown subplot)
        n_assets = return_data.shape[1]
        assert len(fig.data) == n_assets * 2


# ======================================================================
# chart_forward_curves
# ======================================================================

class TestChartForwardCurves:
    @pytest.fixture
    def wide_data(self):
        return rt.data.open_data("dfwide")

    def test_returns_plotly_figure(self, wide_data):
        fig = rt.chart_forward_curves(wide_data, "CL", skip=20)
        assert isinstance(fig, go.Figure)

    def test_with_code_filter(self, wide_data):
        fig = rt.chart_forward_curves(wide_data, "HO", skip=20, yaxis_title="$/g")
        assert isinstance(fig, go.Figure)

    def test_with_cmdty(self, wide_data):
        """Using expiry table for dates."""
        fig = rt.chart_forward_curves(wide_data, "CL", cmdty="cmewti", skip=20)
        assert isinstance(fig, go.Figure)


# ======================================================================
# chart_pairs
# ======================================================================

class TestChartPairs:
    def test_returns_plotly_figure(self):
        df = rt.data.open_data("dfwide")
        df = df[["CL01", "NG01", "HO01"]].dropna()
        fig = rt.chart_pairs(df)
        assert isinstance(fig, go.Figure)

    def test_with_two_columns(self):
        df = rt.data.open_data("dfwide")
        df = df[["CL01", "NG01"]].dropna()
        fig = rt.chart_pairs(df, title="Two Assets")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Two Assets"


# ======================================================================
# dist_desc_plot
# ======================================================================

class TestDistDescPlot:
    def test_runs_without_error(self):
        """dist_desc_plot should produce a matplotlib figure without error."""
        df = rt.data.open_data("dflong")
        x = df["BRN01"].pct_change().dropna()
        # This function creates plt figures but doesn't return one
        rt.dist_desc_plot(x, figsize=(8, 8))
        plt.close("all")
