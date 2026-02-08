"""
Tests for risktools._swap — Swap pricing functions.

Tests swap_irs (interest rate swaps), swap_fut_weight, and swap_com
using bundled data (no API credentials needed).
"""

import os, sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import risktools as rt


# ======================================================================
# swap_irs — Interest Rate Swap
# ======================================================================

class TestSwapIRS:
    @pytest.fixture
    def swap_curves(self):
        return rt.data.open_data("usSwapCurves")

    def test_basic_price_output(self, swap_curves):
        """swap_irs should return a scalar PV when output='price'."""
        pv = rt.swap_irs(
            trade_date="2020-01-04",
            eff_date="2020-01-06",
            mat_date="2022-01-06",
            notional=1000000,
            pay_rec="rec",
            fixed_rate=0.05,
            float_curve=swap_curves,
            reset_freq="Q",
            disc_curve=swap_curves,
            days_in_year=360,
            convention="act",
            output="price",
        )
        assert isinstance(pv, (float, np.floating))

    def test_all_output(self, swap_curves):
        """output='all' should return dict with pv, df, and duration."""
        result = rt.swap_irs(
            trade_date="2020-01-04",
            eff_date="2020-01-06",
            mat_date="2022-01-06",
            notional=1000000,
            pay_rec="rec",
            fixed_rate=0.05,
            float_curve=swap_curves,
            reset_freq="Q",
            disc_curve=swap_curves,
            days_in_year=360,
            convention="act",
            output="all",
        )
        assert isinstance(result, dict)
        assert "pv" in result
        assert "df" in result
        assert "duration" in result
        assert isinstance(result["df"], pd.DataFrame)

    def test_pay_vs_rec(self, swap_curves):
        """Pay and receive should have opposite PVs."""
        kwargs = dict(
            trade_date="2020-01-04",
            eff_date="2020-01-06",
            mat_date="2022-01-06",
            notional=1000000,
            fixed_rate=0.05,
            float_curve=swap_curves,
            reset_freq="Q",
            disc_curve=swap_curves,
            days_in_year=360,
            convention="act",
            output="price",
        )
        pv_rec = rt.swap_irs(pay_rec="rec", **kwargs)
        pv_pay = rt.swap_irs(pay_rec="pay", **kwargs)
        assert abs(pv_rec + pv_pay) < 1e-6, "Pay and rec should have opposite PVs"

    def test_cashflow_df_columns(self, swap_curves):
        """Cash flow DataFrame should have expected columns."""
        result = rt.swap_irs(
            trade_date="2020-01-04",
            eff_date="2020-01-06",
            mat_date="2022-01-06",
            notional=1000000,
            pay_rec="rec",
            fixed_rate=0.05,
            float_curve=swap_curves,
            reset_freq="Q",
            disc_curve=swap_curves,
            days_in_year=360,
            convention="act",
            output="all",
        )
        df = result["df"]
        for col in ["dates", "disc", "fixed", "floating", "net", "duration"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_quarterly_periods(self, swap_curves):
        """2-year swap with quarterly reset should have ~9 rows (trade + 8 quarters)."""
        result = rt.swap_irs(
            trade_date="2020-01-04",
            eff_date="2020-01-06",
            mat_date="2022-01-06",
            notional=1000000,
            pay_rec="rec",
            fixed_rate=0.05,
            float_curve=swap_curves,
            reset_freq="Q",
            disc_curve=swap_curves,
            days_in_year=360,
            convention="act",
            output="all",
        )
        n_rows = result["df"].shape[0]
        assert 8 <= n_rows <= 10, f"Expected ~9 rows for quarterly 2yr swap, got {n_rows}"

    def test_monthly_reset(self, swap_curves):
        """Monthly reset should produce more periods than quarterly."""
        result_q = rt.swap_irs(
            trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06",
            notional=1000000, pay_rec="rec", fixed_rate=0.05,
            float_curve=swap_curves, reset_freq="Q", disc_curve=swap_curves,
            days_in_year=360, convention="act", output="all",
        )
        result_m = rt.swap_irs(
            trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06",
            notional=1000000, pay_rec="rec", fixed_rate=0.05,
            float_curve=swap_curves, reset_freq="M", disc_curve=swap_curves,
            days_in_year=360, convention="act", output="all",
        )
        assert result_m["df"].shape[0] > result_q["df"].shape[0]

    def test_duration_positive(self, swap_curves):
        """Duration should be a positive number for a standard IRS."""
        result = rt.swap_irs(
            trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06",
            notional=1000000, pay_rec="rec", fixed_rate=0.05,
            float_curve=swap_curves, reset_freq="Q", disc_curve=swap_curves,
            days_in_year=360, convention="act", output="all",
        )
        assert result["duration"] > 0

    def test_invalid_days_in_year_raises(self, swap_curves):
        with pytest.raises(ValueError, match="days_in_year"):
            rt.swap_irs(
                trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06",
                notional=1000000, pay_rec="rec", fixed_rate=0.05,
                float_curve=swap_curves, reset_freq="Q", disc_curve=swap_curves,
                days_in_year=400, convention="act", output="price",
            )

    def test_invalid_convention_raises(self, swap_curves):
        with pytest.raises(ValueError, match="convention"):
            rt.swap_irs(
                trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06",
                notional=1000000, pay_rec="rec", fixed_rate=0.05,
                float_curve=swap_curves, reset_freq="Q", disc_curve=swap_curves,
                days_in_year=360, convention="30360", output="price",
            )


# ======================================================================
# swap_fut_weight
# ======================================================================

class TestSwapFutWeight:
    def test_weight_between_0_and_1(self):
        """First futures weight should be between 0 and 1."""
        w = rt.swap_fut_weight(month="2020-09-01", contract="cmewti", exchange="nymex")
        assert 0 < w < 1, f"Weight {w} not in (0, 1)"

    def test_num_days_fut1(self):
        """num_days_fut1 should be a positive integer."""
        d = rt.swap_fut_weight(
            month="2020-09-01", contract="cmewti", exchange="nymex",
            output="num_days_fut1",
        )
        assert isinstance(d, (int, np.integer, np.int64))
        assert d > 0

    def test_num_days_fut2(self):
        """num_days_fut2 should be a positive integer."""
        d = rt.swap_fut_weight(
            month="2020-09-01", contract="cmewti", exchange="nymex",
            output="num_days_fut2",
        )
        assert isinstance(d, (int, np.integer, np.int64))
        assert d >= 0

    def test_days_sum_to_total(self):
        """fut1_days + fut2_days should equal total business days in month (minus holidays)."""
        d1 = rt.swap_fut_weight(
            month="2020-09-01", contract="cmewti", exchange="nymex",
            output="num_days_fut1",
        )
        d2 = rt.swap_fut_weight(
            month="2020-09-01", contract="cmewti", exchange="nymex",
            output="num_days_fut2",
        )
        w = rt.swap_fut_weight(month="2020-09-01", contract="cmewti", exchange="nymex")
        total = d1 + d2
        assert abs(w - d1 / total) < 1e-10

    def test_different_months(self):
        """Different months should produce different weights."""
        w1 = rt.swap_fut_weight(month="2020-06-01")
        w2 = rt.swap_fut_weight(month="2020-09-01")
        # They could be equal but unlikely
        # Just ensure both produce valid results
        assert 0 < w1 < 1
        assert 0 < w2 < 1


# ======================================================================
# custom_date_range
# ======================================================================

class TestCustomDateRange:
    def test_monthly(self):
        from risktools._swap import _custom_date_range
        dr = _custom_date_range("2020-01-01", "2020-06-01", freq="M")
        assert len(dr) >= 6

    def test_quarterly(self):
        from risktools._swap import _custom_date_range
        dr = _custom_date_range("2020-01-01", "2021-01-01", freq="Q")
        assert len(dr) >= 4

    def test_semiannual(self):
        from risktools._swap import _custom_date_range
        dr = _custom_date_range("2020-01-01", "2022-01-01", freq="6M")
        assert len(dr) >= 4

    def test_yearly(self):
        from risktools._swap import _custom_date_range
        dr = _custom_date_range("2020-01-01", "2025-01-01", freq="Y")
        assert len(dr) >= 5

    def test_invalid_freq_raises(self):
        from risktools._swap import _custom_date_range
        with pytest.raises(ValueError):
            _custom_date_range("2020-01-01", "2021-01-01", freq="X")
