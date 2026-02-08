"""Tests for _options.py — Black-Scholes, Black76, Greeks, implied vol, LSM."""

import numpy as np
import pytest
from scipy.stats import norm

import risktools as rt


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
class TestBlackScholes:
    def test_atm_call_known_value(self):
        """ATM call: S=100, K=100, T=1, r=0.05, σ=0.20 → ~10.4506."""
        price = rt.black_scholes(100, 100, 1, 0.05, 0.2, "call")
        assert abs(price - 10.4506) < 0.001

    def test_atm_put_known_value(self):
        price = rt.black_scholes(100, 100, 1, 0.05, 0.2, "put")
        assert abs(price - 5.5735) < 0.001

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT)."""
        S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
        call = rt.black_scholes(S, K, T, r, sigma, "call")
        put = rt.black_scholes(S, K, T, r, sigma, "put")
        parity = call - put - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*e^(-rT)."""
        price = rt.black_scholes(200, 50, 1, 0.05, 0.2, "call")
        intrinsic = 200 - 50 * np.exp(-0.05)
        assert abs(price - intrinsic) < 0.5

    def test_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        price = rt.black_scholes(50, 200, 1, 0.05, 0.2, "call")
        assert price < 0.01

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            rt.black_scholes(100, 100, 1, 0.05, 0.2, "forward")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------
class TestGreeks:
    def test_call_delta_range(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        assert 0 < g["delta"] < 1

    def test_put_delta_range(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "put")
        assert -1 < g["delta"] < 0

    def test_call_put_delta_relationship(self):
        """delta_call - delta_put = 1."""
        gc = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        gp = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "put")
        assert abs(gc["delta"] - gp["delta"] - 1) < 1e-10

    def test_gamma_same_for_call_put(self):
        gc = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        gp = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "put")
        assert abs(gc["gamma"] - gp["gamma"]) < 1e-10

    def test_vega_same_for_call_put(self):
        gc = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        gp = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "put")
        assert abs(gc["vega"] - gp["vega"]) < 1e-10

    def test_gamma_positive(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["vega"] > 0

    def test_call_rho_positive(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        assert g["rho"] > 0

    def test_put_rho_negative(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "put")
        assert g["rho"] < 0

    def test_known_delta(self):
        g = rt.black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
        assert abs(g["delta"] - 0.6368) < 0.001


# ---------------------------------------------------------------------------
# Implied Vol
# ---------------------------------------------------------------------------
class TestImpliedVol:
    def test_roundtrip(self):
        """Price with σ=0.25, recover σ from the price."""
        price = rt.black_scholes(100, 100, 1, 0.05, 0.25)
        iv = rt.implied_vol(price, 100, 100, 1, 0.05)
        assert abs(iv - 0.25) < 1e-6

    def test_roundtrip_put(self):
        price = rt.black_scholes(100, 110, 0.5, 0.03, 0.30, "put")
        iv = rt.implied_vol(price, 100, 110, 0.5, 0.03, "put")
        assert abs(iv - 0.30) < 1e-6

    def test_roundtrip_various_strikes(self):
        for K in [80, 90, 100, 110, 120]:
            price = rt.black_scholes(100, K, 1, 0.05, 0.20)
            iv = rt.implied_vol(price, 100, K, 1, 0.05)
            assert abs(iv - 0.20) < 1e-5


# ---------------------------------------------------------------------------
# Black76
# ---------------------------------------------------------------------------
class TestBlack76:
    def test_atm_put_call_parity(self):
        """C - P = D*(F - K). At the money (F=K), should be 0."""
        F, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        c = rt.black76(F, K, T, r, sigma, "call")
        p = rt.black76(F, K, T, r, sigma, "put")
        assert abs(c - p) < 1e-10

    def test_otm_call_parity(self):
        F, K, T, r, sigma = 100, 110, 0.5, 0.03, 0.25
        c = rt.black76(F, K, T, r, sigma, "call")
        p = rt.black76(F, K, T, r, sigma, "put")
        D = np.exp(-r * T)
        assert abs(c - p - D * (F - K)) < 1e-10

    def test_greeks_keys(self):
        g = rt.black76_greeks(100, 100, 1, 0.05, 0.2)
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_gamma_positive(self):
        g = rt.black76_greeks(100, 100, 1, 0.05, 0.2)
        assert g["gamma"] > 0


# ---------------------------------------------------------------------------
# American Option (LSM)
# ---------------------------------------------------------------------------
class TestAmericanLSM:
    def test_american_put_ge_european(self):
        """American put >= European put."""
        eu = rt.black_scholes(100, 100, 1, 0.05, 0.2, "put")
        am = rt.american_option_lsm(
            100, 100, 1, 0.05, 0.2, "put", n_steps=50, n_sims=50000, seed=42
        )
        assert am >= eu * 0.95  # allow small MC noise

    def test_deep_itm_put(self):
        """Deep ITM American put ≈ K - S (immediate exercise)."""
        am = rt.american_option_lsm(
            50, 100, 0.5, 0.05, 0.2, "put", n_steps=50, n_sims=50000, seed=42
        )
        assert am > 45  # close to 50 intrinsic

    def test_reproducibility(self):
        v1 = rt.american_option_lsm(100, 100, 1, 0.05, 0.2, seed=42)
        v2 = rt.american_option_lsm(100, 100, 1, 0.05, 0.2, seed=42)
        assert v1 == v2


# ---------------------------------------------------------------------------
# Option Payoff
# ---------------------------------------------------------------------------
class TestOptionPayoff:
    def test_call_payoff(self):
        S = np.array([80, 90, 100, 110, 120])
        payoff = rt.option_payoff(S, K=100, option_type="call", premium=5)
        expected = np.array([-5, -5, -5, 5, 15])
        np.testing.assert_array_almost_equal(payoff, expected)

    def test_put_payoff(self):
        S = np.array([80, 90, 100, 110, 120])
        payoff = rt.option_payoff(S, K=100, option_type="put", premium=5)
        expected = np.array([15, 5, -5, -5, -5])
        np.testing.assert_array_almost_equal(payoff, expected)

    def test_scalar_input(self):
        p = rt.option_payoff(110, K=100, option_type="call")
        assert isinstance(p, float)
        assert p == 10.0
