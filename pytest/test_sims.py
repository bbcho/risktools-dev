import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

dir = os.path.dirname(os.path.realpath(__file__))

import risktools as rt

def test_simGBM():
    eps = pd.read_csv('./pytest/data/diffusion.csv', header=None)

    df = rt.simGBM(s0=10, mu=0.0, sigma=0.2, r=0.05, T=1, dt=1/252, sims=20, eps=eps).round(2)

    act = pd.read_csv('./pytest/data/simGBM_output.csv')
    act = act.drop('t', axis=1).T.reset_index(drop=True).T.round(2)

    assert df.equals(act), "simGBM RTL eps failed"

    np.random.seed(123)
    df = rt.simGBM(s0=10, mu=0.0, sigma=0.2, r=0.05, T=1, dt=1/252, sims=20).astype('float').round(4)
    # df.to_csv('./pytest/data/simGBM_output_no_eps.csv', index=False)

    act = pd.read_csv('./pytest/data/simGBM_output_no_eps.csv').astype('float').round(4)
    act = act.T.reset_index(drop=True).T

    assert df.equals(act), "simGBM generated eps failed"


def test_simOU():
    s0 = 5
    mu = 4
    theta = 2
    dt = 0.25
    T = 4
    sigma = 0.32

    eps = np.array([0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,0.5])
    eps = np.c_[eps, eps]

    ans = np.array([
        5.00000,4.50320,4.20680,4.17060,4.12050,
        3.98345,3.93093,3.98466,3.99553,3.95297,
        4.04368,4.05704,3.95172,3.91506,3.97673,
        3.99157,4.06298
        ])
    ans = pd.DataFrame(np.c_[ans, ans])

    # test using dummy eps in both C and Python
    df = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, c=True)
    df = df.T.reset_index(drop=True).T.reset_index(drop=True).round(5)
    ans = ans.T.reset_index(drop=True).T.reset_index(drop=True)
    
    assert np.allclose(df, ans), "C eps test failed"

    df = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, c=False)
    df = df.T.reset_index(drop=True).T.reset_index(drop=True).round(5)

    assert np.allclose(df, ans), "Py eps test failed"

    # test using eps generator

    from numpy.random import default_rng
    rng = default_rng(seed=12345)
    eps = pd.DataFrame(rng.normal(0,1,size=(16, 2)))

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, c=False)
    df2 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, c=False, seed=12345)
    assert np.allclose(df1, df2), "Py gen eps test failed"

    rng = default_rng(seed=12345)
    eps = rng.normal(0,1,size=17*2)
    eps = eps.reshape((2,17)).T
    eps = pd.DataFrame(eps).iloc[1:,:]

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, c=True)
    df2 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, c=True, seed=12345)
    assert np.allclose(df1, df2), "Py gen eps test failed"



if __name__ == "__main__":
    test_simOU()


