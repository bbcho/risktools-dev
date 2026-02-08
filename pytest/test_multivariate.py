import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

dir = os.path.dirname(os.path.realpath(__file__))

import risktools as rt
from numpy.random import Generator, SFC64


def test_calc_spread_MV():

    df = pd.DataFrame(
        dict(
            A=[1, 2, 3, 4, 5],
            B=[2, 3, 4, 5, 6],
        )
    )

    ans = rt.calc_spread_mv(df, {"spread": "A-B"})

    assert ans["spread"].tolist() == [-1, -1, -1, -1, -1], "Spread calculation failed"


def test_fitOU_MV():
    mu = 4
    s0 = 5
    theta = 25
    sigma = 0.32
    T = 1
    dt = 1 / 252

    mm = "OLS"
    df = rt.sim_ou(
        s0=s0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        T=T,
        dt=dt,
        sims=5,
        seed=42,
        log_price=False,
        c=True,
    )
    mu_avg = 0
    theta_avg = 0
    sigma_avg = 0

    for i in range(df.shape[1]):
        params = rt.fit_ou(df.iloc[:, i], dt=dt, method=mm)
        print(params)
        assert np.allclose(
            [*params.values()], [theta, mu, sigma], rtol=0.2
        ), f"{mm} OU MV fit failed"


def test_generate_eps_MV():
    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2
    mu = np.zeros(2)

    sims = 10

    T = 10
    dt = 1 / 252

    eps = rt.generate_eps_mv(cor, T, dt, sims, mu, seed=12345)

    df = pd.DataFrame()
    df["eps1"] = eps[:, 0, 0]
    df["eps2"] = eps[:, 0, 1]

    print(df.corr())

    assert df.corr().iloc[1, 0].round(1) == 0.2, "Correlation failed"
    assert df.corr().iloc[0, 0].round(1) == 1, "Correlation failed"


def test_simOU_MV_logic():
    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4
    sigma = [0.32] * 2

    eps = np.array(
        [
            0.1,
            -0.2,
            0.5,
            0.3,
            -0.4,
            -0.3,
            0.2,
            0.1,
            -0.2,
            0.5,
            0.3,
            -0.4,
            -0.3,
            0.2,
            0.1,
            0.5,
        ]
    )
    eps = np.c_[eps, eps]
    eps = np.stack((eps, eps), axis=2)

    ans = np.array(
        [
            5.00000,
            4.50320,
            4.20680,
            4.17060,
            4.12050,
            3.98345,
            3.93093,
            3.98466,
            3.99553,
            3.95297,
            4.04368,
            4.05704,
            3.95172,
            3.91506,
            3.97673,
            3.99157,
            4.06298,
        ]
    )
    ans = np.c_[ans, ans]
    ans = np.stack((ans, ans), axis=2)

    # test using dummy eps
    df = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, sigma=sigma, eps=eps, log_price=True
    )

    assert np.allclose(df, ans), "OU MV simulation failed"


def test_simOU_MV_eps():
    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4
    sigma = [0.32] * 2

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    eps = rt.generate_eps_mv(cor=cor, T=T, dt=dt, sims=2, seed=12345)
    print(eps)

    df1 = rt.sim_ou_mv(s0=s0, mu=mu, theta=theta, T=T, sigma=sigma, eps=eps)
    df2 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345
    )

    assert np.allclose(df1, df2), "Seed eps test failed"


def test_simOU_MV_mu():
    s0 = [5] * 2
    mu = [4] * 2
    sigma = [0.32] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4

    N = int(T / dt)
    mus = np.ones((N, 2)) * mu[0]

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    df1 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345
    )
    df2 = rt.sim_ou_mv(
        s0=s0, mu=mus, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345
    )
    assert np.allclose(df1, df2), "Time varying mu test failed"


def test_simOU_MV_sigma():

    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4

    N = int(T / dt)
    sigma = [0.32] * 2
    sigmas = np.ones((N, 2)) * sigma[0]

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    df1 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345
    )
    df2 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigmas, cor=cor, sims=2, seed=12345
    )
    assert np.allclose(df1, df2), "Time varying sigma test failed"

    sigmas = np.ones((N, 3, 2)) * sigma[0]
    df1 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=3, seed=12345
    )
    df2 = rt.sim_ou_mv(
        s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigmas, cor=cor, sims=3, seed=12345
    )
    assert np.allclose(df1, df2), "Time varying sigma test failed"


def test_simOUJ_MV_logic():
    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4
    sims = 2
    sigma = [0.32] * 2

    jump_avgsize = [1] * 2
    jump_prob = [0.1] * 2
    jump_stdv = [0.32] * 2

    # fmt: off
    eps = np.array([0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,0.5])
    eps = np.c_[eps, eps]
    eps = np.stack((eps,eps), axis=2)

    elp = np.ones(eps.shape)
    ejp = np.zeros(eps.shape)
    ejp[5,:,:] = 1

    ans = np.array([
        5.00000,2.33000,4.08449,4.03448,3.95686,3.59113,
        4.97335,2.46342,4.27228,3.34032,4.54230,3.30157,
        4.07815,3.51914,4.30190,3.50626,4.47704,
    ])
    ans = np.c_[ans, ans]
    ans = np.stack((ans,ans), axis=2)

    # fmt: on

    # test using dummy eps
    df = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        eps=eps,
        elp=elp,
        ejp=ejp,
        sims=sims,
    )

    assert np.allclose(df, ans), "OUJ MV simulation failed"


def test_simOUJ_MV_eps():
    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4
    sigma = [0.32] * 2

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    sims = 2

    jump_avgsize = [1] * 2
    jump_prob = [0.1] * 2
    jump_stdv = [0.32] * 2

    eps = rt.generate_eps_mv(cor=cor, T=T, dt=dt, sims=2, seed=12345)

    df1 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        eps=eps,
        sims=sims,
        seed=12345,
    )
    df2 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=sims,
        cor=cor,
        seed=12345,
    )

    assert np.allclose(df1, df2), "Seed eps test failed"


def test_simOUJ_MV_mu():
    s0 = [5] * 2
    mu = [4] * 2
    sigma = [0.32] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4

    N = int(T / dt)
    mus = np.ones((N, 2)) * mu[0]

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    sims = 2

    jump_avgsize = [1] * 2
    jump_prob = [0.1] * 2
    jump_stdv = [0.32] * 2

    df1 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=2,
        seed=12345,
    )
    df2 = rt.sim_ouj_mv(
        s0=s0,
        mu=mus,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=2,
        seed=12345,
    )
    assert np.allclose(df1, df2), "Time varying mu test failed"


def test_simOUJ_MV_sigma():

    s0 = [5] * 2
    mu = [4] * 2
    theta = [2] * 2
    dt = 0.25
    T = 4

    N = int(T / dt)
    sigma = [0.32] * 2
    sigmas = np.ones((N, 2)) * sigma[0]

    cor = np.diag(np.ones(2))
    cor[1, 0] = 0.2
    cor[0, 1] = 0.2

    sims = 2

    jump_avgsize = [1] * 2
    jump_prob = [0.1] * 2
    jump_stdv = [0.32] * 2

    df1 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=2,
        seed=12345,
    )
    df2 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigmas,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=2,
        seed=12345,
    )
    assert np.allclose(df1, df2), "Time varying sigma test failed"

    sigmas = np.ones((N, 3, 2)) * sigma[0]
    df1 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigma,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=3,
        seed=12345,
    )
    df2 = rt.sim_ouj_mv(
        s0=s0,
        mu=mu,
        theta=theta,
        T=T,
        dt=dt,
        sigma=sigmas,
        cor=cor,
        jump_avgsize=jump_avgsize,
        jump_prob=jump_prob,
        jump_stdv=jump_stdv,
        sims=3,
        seed=12345,
    )
    assert np.allclose(df1, df2), "Time varying sigma test failed"


# ======================================================================
# generate_random_portfolio_weights
# ======================================================================


def test_generate_random_portfolio_weights_shape():
    """generate_random_portfolio_weights should return correct shape."""
    weights = rt.generate_random_portfolio_weights(3, 50)
    assert weights.shape == (50, 3)


def test_generate_random_portfolio_weights_sum_to_one():
    """Each row of weights should sum to 1."""
    weights = rt.generate_random_portfolio_weights(5, 100)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_generate_random_portfolio_weights_non_negative():
    """All weights should be >= 0 (derived from uniform distribution)."""
    weights = rt.generate_random_portfolio_weights(4, 200)
    assert (weights >= 0).all()


def test_generate_random_portfolio_weights_defaults():
    """Default number_sims should be 2500."""
    weights = rt.generate_random_portfolio_weights(2)
    assert weights.shape == (2500, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)


# ======================================================================
# calculate_payoffs
# ======================================================================


def test_calculate_payoffs_default_strike_zero():
    """With default strike=0 on GBM (always positive), payoffs = final prices."""
    sims = rt.sim_gbm_mv(
        s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
        cor=[[1, 0.5], [0.5, 1]], sims=20, seed=42,
    )
    payoffs = rt.calculate_payoffs(sims)
    assert payoffs.shape == (20, 2)
    # GBM prices are always positive, so clip(x - 0, 0, None) = x
    assert np.allclose(payoffs, sims[-1, :, :])


def test_calculate_payoffs_with_strike():
    """calculate_payoffs with strike > 0 clips values below strike to 0."""
    # Shape (2, 3, 2): 2 time steps, 3 sims, 2 assets
    arr = np.array([
        [[80, 90], [110, 120], [100, 100]],
        [[85, 95], [115, 125], [95, 105]],
    ], dtype=float)
    payoffs = rt.calculate_payoffs(arr, strike=100)
    # final row: [[85, 95], [115, 125], [95, 105]]
    # clipped to max(x - 100, 0): [[0, 0], [15, 25], [0, 5]]
    expected = np.array([[0, 0], [15, 25], [0, 5]], dtype=float)
    assert np.allclose(payoffs, expected)


def test_calculate_payoffs_custom_func():
    """calculate_payoffs with custom payoff functions."""
    arr = np.ones((5, 3, 2))  # 5 steps, 3 sims, 2 assets
    arr[:, :, 0] *= 2.0
    arr[:, :, 1] *= 3.0

    # payoff = sum along time axis
    def sum_payoff(x):
        return x.sum(axis=0)

    payoffs = rt.calculate_payoffs(arr, payoff_funcs=[sum_payoff, sum_payoff])
    assert payoffs.shape == (3, 2)
    assert np.allclose(payoffs[:, 0], 10.0)  # 2.0 * 5 steps
    assert np.allclose(payoffs[:, 1], 15.0)  # 3.0 * 5 steps


def test_calculate_payoffs_wrong_num_funcs():
    """Passing wrong number of payoff functions should raise ValueError."""
    arr = np.ones((5, 3, 2))

    def dummy(x):
        return x.sum(axis=0)

    with pytest.raises(ValueError, match="payoff function for each asset"):
        rt.calculate_payoffs(arr, payoff_funcs=[dummy])


# ======================================================================
# simulate_efficient_frontier
# ======================================================================


def test_simulate_efficient_frontier_shape():
    """simulate_efficient_frontier returns (n_portfolios, 2)."""
    payoffs = np.random.RandomState(42).randn(50, 3) + 100
    weights = rt.generate_random_portfolio_weights(3, 80)
    ef = rt.simulate_efficient_frontier(payoffs, weights)
    assert ef.shape == (80, 2)


def test_simulate_efficient_frontier_known_values():
    """Check risk (col 0) and return (col 1) for a known portfolio."""
    payoffs = np.array([[10.0, 20.0], [30.0, 40.0]])
    weights = np.array([[0.5, 0.5]])
    ef = rt.simulate_efficient_frontier(payoffs, weights)
    # portfolio returns per sim: [10*0.5+20*0.5, 30*0.5+40*0.5] = [15, 35]
    expected_mean = np.mean([15.0, 35.0])
    expected_std = np.std([15.0, 35.0])
    assert ef[0, 0] == pytest.approx(expected_std)
    assert ef[0, 1] == pytest.approx(expected_mean)


# ======================================================================
# make_efficient_frontier_table
# ======================================================================


def test_make_efficient_frontier_table_columns():
    """make_efficient_frontier_table returns DataFrame with correct columns."""
    returns = np.array([[0.1, 0.05], [0.2, 0.08]])
    weights = np.array([[0.6, 0.4], [0.3, 0.7]])
    table = rt.make_efficient_frontier_table(returns, weights, asset_names=["StockA", "StockB"])
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["Risk", "Expected Return", "StockA", "StockB"]
    assert table.shape == (2, 4)


def test_make_efficient_frontier_table_values():
    """Values in the table should match the input arrays."""
    returns = np.array([[0.1, 0.05], [0.2, 0.08]])
    weights = np.array([[0.6, 0.4], [0.3, 0.7]])
    table = rt.make_efficient_frontier_table(returns, weights, asset_names=["A", "B"])
    assert table["Risk"].iloc[0] == pytest.approx(0.1)
    assert table["Expected Return"].iloc[0] == pytest.approx(0.05)
    assert table["A"].iloc[0] == pytest.approx(0.6)
    assert table["B"].iloc[0] == pytest.approx(0.4)


def test_make_efficient_frontier_table_default_names():
    """When asset_names is None, column names should be integers 1..N."""
    returns = np.array([[0.1, 0.05]])
    weights = np.array([[0.6, 0.4]])
    table = rt.make_efficient_frontier_table(returns, weights)
    assert list(table.columns) == ["Risk", "Expected Return", 1, 2]


# ======================================================================
# efficient frontier end-to-end pipeline
# ======================================================================


def test_efficient_frontier_pipeline():
    """End-to-end: sim_gbm_mv -> calculate_payoffs -> weights -> frontier -> table."""
    sims = rt.sim_gbm_mv(
        s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
        cor=[[1, 0.5], [0.5, 1]], sims=50, seed=42,
    )
    payoffs = rt.calculate_payoffs(sims)
    assert payoffs.shape == (50, 2)

    weights = rt.generate_random_portfolio_weights(2, 100)
    assert weights.shape == (100, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all()

    ef = rt.simulate_efficient_frontier(payoffs, weights)
    assert ef.shape == (100, 2)

    table = rt.make_efficient_frontier_table(ef, weights, asset_names=["A", "B"])
    assert isinstance(table, pd.DataFrame)
    assert "A" in table.columns
    assert "B" in table.columns
    assert "Risk" in table.columns
    assert "Expected Return" in table.columns
    assert table.shape == (100, 4)


# ======================================================================
# MvGbm class
# ======================================================================


class TestMvGbm:
    def test_simulate_and_output(self):
        """MvGbm should simulate and produce a multi-indexed DataFrame."""
        mvgbm = rt.MvGbm(
            s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
            cor=[[1, 0.5], [0.5, 1]], asset_names=["A", "B"],
        )
        mvgbm.simulate(sims=10, seed=42)
        assert mvgbm.sims.shape == (253, 10, 2)
        df = mvgbm.output()
        assert isinstance(df, pd.DataFrame)
        # output() generates default names when prices is None
        assert set(df.index.get_level_values("asset").unique()) == {"Asset 0", "Asset 1"}

    def test_initial_values(self):
        """First row of sims should match s0."""
        mvgbm = rt.MvGbm(
            s0=[50, 200], r=0.0, sigma=[0.1, 0.1], T=1, dt=1 / 252,
            cor=[[1, 0], [0, 1]],
        )
        mvgbm.simulate(sims=5, seed=42)
        assert mvgbm.sims[0, 0, 0] == pytest.approx(50)
        assert mvgbm.sims[0, 0, 1] == pytest.approx(200)

    def test_all_sims_start_at_s0(self):
        """All simulations across all sims should start at s0."""
        mvgbm = rt.MvGbm(
            s0=[100, 200, 300], r=0.03, sigma=[0.1, 0.2, 0.3], T=0.5, dt=1 / 252,
            cor=np.eye(3),
        )
        mvgbm.simulate(sims=20, seed=99)
        for sim_idx in range(20):
            assert mvgbm.sims[0, sim_idx, 0] == pytest.approx(100)
            assert mvgbm.sims[0, sim_idx, 1] == pytest.approx(200)
            assert mvgbm.sims[0, sim_idx, 2] == pytest.approx(300)

    def test_shape_three_assets(self):
        """Check shape for 3-asset portfolio with T=2."""
        mvgbm = rt.MvGbm(
            s0=[100, 100, 100], r=0.01, sigma=[0.1, 0.1, 0.1], T=2, dt=1 / 252,
            cor=[[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]],
        )
        mvgbm.simulate(sims=15, seed=42)
        expected_steps = int(2 / (1 / 252)) + 1  # 505
        assert mvgbm.sims.shape == (expected_steps, 15, 3)

    def test_positive_prices(self):
        """GBM prices should always be strictly positive."""
        mvgbm = rt.MvGbm(
            s0=[100, 100], r=0.0, sigma=[0.5, 0.5], T=1, dt=1 / 252,
            cor=[[1, 0.9], [0.9, 1]],
        )
        mvgbm.simulate(sims=50, seed=42)
        assert (mvgbm.sims > 0).all()

    def test_seed_reproducibility(self):
        """Same seed should produce identical simulations."""
        kwargs = dict(
            s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
            cor=[[1, 0.5], [0.5, 1]],
        )
        mvgbm1 = rt.MvGbm(**kwargs)
        mvgbm1.simulate(sims=10, seed=42)
        mvgbm2 = rt.MvGbm(**kwargs)
        mvgbm2.simulate(sims=10, seed=42)
        assert np.allclose(mvgbm1.sims, mvgbm2.sims)

    def test_output_default_names(self):
        """Output without explicit names uses 'Asset 0', 'Asset 1', etc."""
        mvgbm = rt.MvGbm(
            s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
            cor=[[1, 0], [0, 1]],
        )
        mvgbm.simulate(sims=5, seed=42)
        df = mvgbm.output()
        assert set(df.index.get_level_values("asset").unique()) == {"Asset 0", "Asset 1"}

    def test_output_has_date_index(self):
        """Output DataFrame should have 'date' as an index level."""
        mvgbm = rt.MvGbm(
            s0=[100, 100], r=0.05, sigma=[0.2, 0.3], T=1, dt=1 / 252,
            cor=[[1, 0], [0, 1]],
        )
        mvgbm.simulate(sims=3, seed=42)
        df = mvgbm.output()
        assert "date" in df.index.names
        assert "asset" in df.index.names

    def test_requires_params_when_no_prices(self):
        """MvGbm should raise ValueError if s0/sigma/cor not provided and no prices."""
        with pytest.raises(ValueError):
            rt.MvGbm(r=0.05, T=1, dt=1 / 252, s0=[100], sigma=None, cor=None)


# ======================================================================
# MvOu class
# ======================================================================


class TestMvOu:
    def test_simulate_and_output(self):
        """MvOu should simulate and produce a multi-indexed DataFrame."""
        mvou = rt.MvOu(
            s0=[5, 5], mu=[4, 4], theta=[2, 2], sigma=[0.5, 0.5],
            T=1, dt=1 / 252, cor=[[1, 0.5], [0.5, 1]], asset_names=["X", "Y"],
        )
        mvou.fit()
        mvou.simulate(sims=10, seed=42)
        assert mvou.sims.shape == (253, 10, 2)
        df = mvou.output()
        assert isinstance(df, pd.DataFrame)
        # output() generates default names when prices is None
        assert set(df.index.get_level_values("asset").unique()) == {"Asset 0", "Asset 1"}

    def test_mean_reversion(self):
        """OU process should revert to mu over long time horizon with many sims."""
        mvou = rt.MvOu(
            s0=[10, 10], mu=[5, 5], theta=[3, 3], sigma=[0.3, 0.3],
            T=5, dt=1 / 252, cor=[[1, 0], [0, 1]],
        )
        mvou.fit()
        mvou.simulate(sims=500, seed=42)
        final_means = mvou.sims[-1, :, :].mean(axis=0)
        assert final_means[0] == pytest.approx(5, abs=0.5)
        assert final_means[1] == pytest.approx(5, abs=0.5)

    def test_initial_values(self):
        """First row of sims should match s0."""
        mvou = rt.MvOu(
            s0=[3, 7], mu=[5, 5], theta=[2, 2], sigma=[0.5, 0.5],
            T=1, dt=1 / 252, cor=[[1, 0], [0, 1]],
        )
        mvou.fit()
        mvou.simulate(sims=5, seed=42)
        for sim_idx in range(5):
            assert mvou.sims[0, sim_idx, 0] == pytest.approx(3)
            assert mvou.sims[0, sim_idx, 1] == pytest.approx(7)

    def test_seed_reproducibility(self):
        """Same seed should produce identical simulations."""
        kwargs = dict(
            s0=[5, 5], mu=[4, 4], theta=[2, 2], sigma=[0.5, 0.5],
            T=1, dt=1 / 252, cor=[[1, 0.5], [0.5, 1]],
        )
        mvou1 = rt.MvOu(**kwargs)
        mvou1.fit()
        mvou1.simulate(sims=10, seed=42)
        mvou2 = rt.MvOu(**kwargs)
        mvou2.fit()
        mvou2.simulate(sims=10, seed=42)
        assert np.allclose(mvou1.sims, mvou2.sims)

    def test_parameters_property(self):
        """parameters property should include s0, mu, theta, annualized_sigma rows."""
        mvou = rt.MvOu(
            s0=[5, 5], mu=[4, 4], theta=[2, 2], sigma=[0.5, 0.5],
            T=1, dt=1 / 252, cor=[[1, 0], [0, 1]], asset_names=["A", "B"],
        )
        mvou.fit()
        params = mvou.parameters
        assert isinstance(params, pd.DataFrame)
        assert "s0" in params.index
        assert "mu" in params.index
        assert "theta" in params.index
        assert "annualized_sigma" in params.index
        assert list(params.columns) == ["A", "B"]
        assert params.loc["s0", "A"] == pytest.approx(5)
        assert params.loc["mu", "B"] == pytest.approx(4)

    def test_requires_params_when_no_prices(self):
        """MvOu should raise ValueError if required params missing and no prices."""
        with pytest.raises(ValueError):
            rt.MvOu(T=1, dt=1 / 252, s0=[5], mu=None, theta=None, sigma=None, cor=None)

    def test_fit_required_before_simulate(self):
        """simulate() should fail if fit() was not called first (no _params)."""
        mvou = rt.MvOu(
            s0=[5, 5], mu=[4, 4], theta=[2, 2], sigma=[0.5, 0.5],
            T=1, dt=1 / 252, cor=[[1, 0], [0, 1]],
        )
        with pytest.raises((AttributeError, TypeError)):
            mvou.simulate(sims=5, seed=42)


if __name__ == "__main__":
    test_simOUJ_MV_mu()
