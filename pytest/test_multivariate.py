import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

dir = os.path.dirname(os.path.realpath(__file__))

import risktools as rt
from numpy.random import Generator, SFC64

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
    cor = np.diag(np.ones(2))
    cor[1,0] = 0.2
    cor[0,1] = 0.2
    mu = np.zeros(2)

    sims = 10

    T = 10
    dt = 1/252

    eps = rt.generate_eps_MV(cor, T, dt, sims, mu, seed=12345)

    df = pd.DataFrame()
    df['eps1'] = eps[:,0,0]
    df['eps2'] = eps[:,0,1]

    print(df.corr())

    assert df.corr().iloc[1,0].round(1) == 0.2, "Correlation failed"
    assert df.corr().iloc[0,0].round(1) == 1, "Correlation failed"


def test_simOU_MV_logic():
    s0 = [5]*2
    mu = [4]*2
    theta = [2]*2
    dt = 0.25
    T = 4
    sigma = [0.32]*2

    eps = np.array([0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,0.5])
    eps = np.c_[eps, eps]
    eps = np.stack((eps,eps), axis=2)

    ans = np.array([
        5.00000,4.50320,4.20680,4.17060,4.12050,
        3.98345,3.93093,3.98466,3.99553,3.95297,
        4.04368,4.05704,3.95172,3.91506,3.97673,
        3.99157,4.06298
        ])
    ans = np.c_[ans, ans]
    ans = np.stack((ans,ans), axis=2)

    # test using dummy eps
    df = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, sigma=sigma, eps=eps, log_price=True)  
    
    assert np.allclose(df, ans), "OU MV simulation failed"

def test_simOU_MV_eps():
    s0 = [5]*2
    mu = [4]*2
    theta = [2]*2
    dt = 0.25
    T = 4
    sigma = [0.32]*2

    cor = np.diag(np.ones(2))
    cor[1,0] = 0.2
    cor[0,1] = 0.2

    eps = rt.generate_eps_MV(cor=cor, T=T, dt=dt, sims=2, seed=12345)
    print(eps)

    df1 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, sigma=sigma, eps=eps)
    df2 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345)

    assert np.allclose(df1, df2), "Seed eps test failed"

def test_simOU_MV_mu():
    s0 = [5]*2
    mu = [4]*2
    sigma = [0.32]*2
    theta = [2]*2
    dt = 0.25
    T = 4

    N = int(T/dt)
    mus = np.ones((N,2)) * mu[0]
    

    cor = np.diag(np.ones(2))
    cor[1,0] = 0.2
    cor[0,1] = 0.2

    df1 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345)
    df2 = rt.simOU_MV(s0=s0, mu=mus, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345)
    assert np.allclose(df1, df2), "Time varying mu test failed"

def test_simOU_MV_sigma():

    s0 = [5]*2
    mu = [4]*2
    theta = [2]*2
    dt = 0.25
    T = 4

    N = int(T/dt)
    sigma = [0.32]*2
    sigmas = np.ones((N,2)) * sigma[0]

    cor = np.diag(np.ones(2))
    cor[1,0] = 0.2
    cor[0,1] = 0.2

    df1 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=2, seed=12345)
    df2 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigmas, cor=cor, sims=2, seed=12345)
    assert np.allclose(df1, df2), "Time varying sigma test failed"

    sigmas = np.ones((N,3,2)) * sigma[0]
    df1 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigma, cor=cor, sims=3, seed=12345)
    df2 = rt.simOU_MV(s0=s0, mu=mu, theta=theta, T=T, dt=dt, sigma=sigmas, cor=cor, sims=3, seed=12345)
    assert np.allclose(df1, df2), "Time varying sigma test failed"



if __name__ == "__main__":
    test_simOU_MV_sigma()