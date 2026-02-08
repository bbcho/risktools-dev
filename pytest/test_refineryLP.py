"""
Tests for risktools._refineryLP — Refinery LP optimization.

Uses the bundled refinery optimization data and verifies the LP solution
is feasible and optimal.
"""

import os, sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import risktools as rt


class TestRefineryLP:
    @pytest.fixture
    def refinery_data(self):
        data = rt.data.open_data("refineryLPdata")
        return data["inputs"], data["outputs"]

    def test_basic_run(self, refinery_data):
        """refineryLP should return a dict with 'profit' and 'slate'."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products)
        assert isinstance(result, dict)
        assert "profit" in result
        assert "slate" in result

    def test_profit_positive(self, refinery_data):
        """A valid refinery should have positive profit."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products)
        assert result["profit"] > 0, "Refinery profit should be positive"

    def test_slate_nonnegative(self, refinery_data):
        """Crude slate (barrels processed) should be non-negative."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products)
        assert all(x >= -1e-10 for x in result["slate"]), "Slate values should be non-negative"

    def test_return_all(self, refinery_data):
        """return_all=True should return the full scipy OptimizeResult."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products, return_all=True)
        # scipy.optimize.OptimizeResult has .fun, .x, .success attributes
        assert hasattr(result, "fun")
        assert hasattr(result, "x")
        assert result.success, "LP should converge successfully"

    def test_product_constraints_satisfied(self, refinery_data):
        """Product output should not exceed max_prod constraints."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products)
        slate = result["slate"]
        # Check: A_ub @ x <= b_ub
        yields = products[["LightSweet_yield", "HeavySour_yield"]].values
        max_prod = products["max_prod"].values
        actual_prod = yields @ slate
        # Each product output should be <= max_prod (with tolerance)
        assert all(actual_prod <= max_prod + 1e-6), "Product output exceeds max constraints"

    def test_two_crudes(self, refinery_data):
        """Slate should have exactly 2 elements (LightSweet, HeavySour)."""
        crudes, products = refinery_data
        result = rt.refinery_lp(crudes, products)
        assert len(result["slate"]) == 2


class TestRefineryLPEdgeCases:
    def test_zero_product_prices(self):
        """If all product prices are zero and crude costs are negative, profit should be non-positive."""
        crudes = pd.DataFrame({
            "info": ["price", "processing_fee"],
            "LightSweet": [-50.0, -5.0],
            "HeavySour": [-40.0, -4.0],
        })
        products = pd.DataFrame({
            "product": ["gasoline", "diesel", "jet_fuel"],
            "prices": [0.0, 0.0, 0.0],
            "LightSweet_yield": [0.5, 0.3, 0.2],
            "HeavySour_yield": [0.3, 0.4, 0.3],
            "max_prod": [1000, 1000, 1000],
        })
        result = rt.refinery_lp(crudes, products)
        assert result["profit"] <= 1e-6, "Profit should be non-positive with zero product prices and negative costs"
