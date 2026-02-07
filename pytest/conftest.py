"""Shared fixtures for risktools test suite."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import risktools as rt


# ---------------------------------------------------------------------------
# Simple deterministic return series for canonical calculations
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns_simple():
    """10 known daily returns for hand-calculable tests.

    Returns: [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01]
    """
    dates = pd.bdate_range("2020-01-02", periods=10, freq="B")
    vals = [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01]
    return pd.Series(vals, index=dates, name="test_returns")


@pytest.fixture
def monthly_returns():
    """12 monthly returns (one year) for annualization tests."""
    dates = pd.date_range("2020-01-31", periods=12, freq="M")
    vals = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02]
    return pd.Series(vals, index=dates, name="monthly")


@pytest.fixture
def weekly_returns():
    """52 weekly returns for one year of data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
    vals = np.random.normal(0.001, 0.02, 52)
    return pd.Series(vals, index=dates, name="weekly")


@pytest.fixture
def multi_asset_returns():
    """DataFrame with two assets (A and B) of daily returns."""
    dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
    np.random.seed(123)
    a = np.random.normal(0.001, 0.015, 20)
    b = np.random.normal(0.0005, 0.02, 20)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def drawdown_series():
    """A return series with a known drawdown pattern.

    Cumulative: up to 1.10, then down to ~0.935, then recover to ~1.03
    """
    dates = pd.bdate_range("2020-01-02", periods=8, freq="B")
    vals = [0.05, 0.05, -0.10, -0.05, 0.02, 0.03, 0.04, -0.01]
    return pd.Series(vals, index=dates, name="drawdown_test")


@pytest.fixture
def dflong():
    """Load the bundled dflong dataset."""
    return rt.data.open_data("dflong")


@pytest.fixture
def dfwide():
    """Load the bundled dfwide dataset."""
    return rt.data.open_data("dfwide")
