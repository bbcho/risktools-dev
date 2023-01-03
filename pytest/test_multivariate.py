import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

dir = os.path.dirname(os.path.realpath(__file__))

import risktools as rt

def test_calc_spread_MV():

    df = pd.DataFrame(dict(
        A = [1, 2, 3, 4, 5],
        B = [2, 3, 4, 5, 6],
    ))

    ans = rt.calc_spread_MV(df, {'spread': 'A-B'})

    assert ans['spread'].tolist() == [-1, -1, -1, -1, -1], "Spread calculation failed"


def test_fitOU_MV():
    mu = 4
    s0 = 5
    theta = 25
    sigma = 0.32
    T = 1
    dt = 1/252

    mm = 'OLS'
    df = rt.simOU(s0=s0, mu=mu, theta=theta, sigma=sigma, T=T, dt=dt, sims=5, seed=42, log_price=False, c=True)
    mu_avg = 0
    theta_avg = 0
    sigma_avg = 0

    for i in range(df.shape[1]):
        params = rt.fitOU(df.iloc[:,i], dt=dt, method=mm)
        print(params)
        assert np.allclose([*params.values()], [theta, mu, sigma], rtol=0.2), f"{mm} OU MV fit failed"


def test_generate_eps_MV():
    pass