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


def test_simOU_logic():
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
    df = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, log_price=True, c=True)  
    df = df.T.reset_index(drop=True).T.reset_index(drop=True).round(5)
    ans = ans.T.reset_index(drop=True).T.reset_index(drop=True)
    
    assert np.allclose(df, ans), "C eps test failed"

    df = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, log_price=True, c=False) 
    df = df.T.reset_index(drop=True).T.reset_index(drop=True).round(5)

    assert np.allclose(df, ans), "Py eps test failed"

def test_simOU_eps():
    s0 = 5
    mu = 4
    theta = 2
    dt = 0.25
    T = 4
    sigma = 0.32

    # test using eps generator

    from numpy.random import default_rng, Generator, SFC64
    # rng = default_rng(seed=12345)
    rng = Generator(SFC64(seed=12345))
    eps = pd.DataFrame(rng.normal(0,1,size=(16, 2)))

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, log_price=False, c=False) 
    df2 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    assert np.allclose(df1, df2), "Py seed eps test failed"

    # rng = default_rng(seed=12345)
    rng = Generator(SFC64(seed=12345))
    eps = rng.normal(0,1,size=17*2)
    eps = eps.reshape((2,17)).T
    eps = pd.DataFrame(eps).iloc[1:,:]

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, eps=eps, log_price=False, c=True) 
    df2 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    assert np.allclose(df1, df2), "C seed eps test failed"

def test_simOU_mu():
    s0 = 5
    mu = 4
    theta = 2
    dt = 0.25
    T = 4
    sigma = 0.32

    #################################
    # test time varying mu
    #################################
    mus = np.ones(16) * mu
      
    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    df2 = rt.simOU(s0, mus, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    assert np.allclose(df1, df2), "Py time varying mu test failed"

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    df2 = rt.simOU(s0, mus, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    assert np.allclose(df1, df2), "C time varying mu test failed"

def test_simOU_sigma():
    s0 = 5
    mu = 4
    theta = 2
    dt = 0.25
    T = 4
    sigma = 0.32

    #################################
    # test time varying sigma
    #################################
    sigmas = np.ones(16) * sigma

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    df2 = rt.simOU(s0, mu, theta, sigmas, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    assert np.allclose(df1, df2), "Py time varying sigma test failed"

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    df2 = rt.simOU(s0, mu, theta, sigmas, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    assert np.allclose(df1, df2), "C time varying sigma test failed"


    sigmas = np.ones((16,2)) * sigma

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    df2 = rt.simOU(s0, mu, theta, sigmas, T, dt, sims=2, seed=12345, log_price=False, c=False) 
    assert np.allclose(df1, df2), "Py time varying sigma array test failed"

    df1 = rt.simOU(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    df2 = rt.simOU(s0, mu, theta, sigmas, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    assert np.allclose(df1, df2), "C time varying sigma array test failed"

    # test time varying and 2D sigma

    sigma = np.ones((252, 1000)) * 0.1
    sigma[100:,:] = 0.5

    df = rt.simOU(s0, mu, theta, sigma, T=1, dt=1/252, sims=1000, seed=12345, log_price=False, c=False)

    assert df.iloc[152,:].std()/df.iloc[25,:].std() > 3, "Time varying sigma test failed"


def test_simOUJ_logic():

    eps = np.array([0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,-0.2,0.5,0.3,-0.4,-0.3,0.2,0.1,0.5])
    eps = np.c_[eps, eps]
    elp = np.ones(eps.shape)
    ejp = np.zeros(eps.shape)
    ejp[5,:] = 1

    s0=5
    mu=4
    theta=2
    dt=0.25
    sigma=0.32
    T=4
    sims=2
    jump_avgsize=1
    jump_prob=0.1
    jump_stdv=0.32

    df = rt.simOUJ(
        T=T, s0=s0, mu=mu, theta=theta, dt=dt, sigma=sigma, 
        jump_avgsize=jump_avgsize, jump_prob=jump_prob, jump_stdv=jump_stdv, 
        eps=eps, elp=elp, ejp=ejp, sims=sims)

    ans = np.array([
        5.00000,2.33000,4.08449,4.03448,3.95686,3.59113,
        4.97335,2.46342,4.27228,3.34032,4.54230,3.30157,
        4.07815,3.51914,4.30190,3.50626,4.47704,
    ])
    ans = pd.DataFrame(np.c_[ans, ans])

    # test using dummy eps in both C and Python
    df = df.T.reset_index(drop=True).T.reset_index(drop=True).round(5)
    ans = ans.T.reset_index(drop=True).T.reset_index(drop=True)
    
    assert np.allclose(df, ans), "Py eps test failed"


def test_simOUJ_eps():

    s0=5
    mu=4
    theta=2
    dt=0.25
    sigma=0.32
    T=4
    sims=2
    jump_avgsize=1
    jump_prob=0.1
    jump_stdv=0.32
    # test using eps generator

    from numpy.random import default_rng, Generator, SFC64
    rng = Generator(SFC64(seed=12345))
        
    eps = rng.normal(size=(int(T/dt), sims))
    elp = rng.lognormal(mean=np.log(jump_avgsize), sigma=jump_stdv, size=(int(T/dt), sims))
    ejp = rng.poisson(lam=jump_prob * dt, size=(int(T/dt), sims))

    df1 = rt.simOUJ(
        s0=s0, mu=mu, theta=theta, sigma=sigma, T=T, dt=dt, sims=2, 
        eps=eps, elp=elp, ejp=ejp, 
        jump_avgsize=jump_avgsize, jump_prob=jump_prob, jump_stdv=jump_stdv
        )
    df2 = rt.simOUJ(
        s0=s0, mu=mu, theta=theta, sigma=sigma, T=T, dt=dt, sims=2,
        jump_avgsize=jump_avgsize, jump_prob=jump_prob, jump_stdv=jump_stdv, 
        seed=12345
        ) 
    assert np.allclose(df1, df2), "Py seed eps test failed"

def test_simOUJ_mu():
    s0 = 5
    mu = 4
    theta = 2
    dt = 0.25
    T = 4
    sigma = 0.32

    #################################
    # test time varying mu
    #################################
    mus = np.ones(int(T/dt)) * mu
      
    df1 = rt.simOUJ(s0=s0, mu=mu, theta=theta, sigma=sigma, T=T, dt=dt, sims=2, seed=12345) 
    df2 = rt.simOUJ(s0=s0, mu=mus, theta=theta, sigma=sigma, T=T, dt=dt, sims=2, seed=12345) 
    assert np.allclose(df1, df2), "Py time varying mu test failed"

    # df1 = rt.simOUJ(s0, mu, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    # df2 = rt.simOUJ(s0, mus, theta, sigma, T, dt, sims=2, seed=12345, log_price=False, c=True) 
    # assert np.allclose(df1, df2), "C time varying mu test failed"

def test_simOUJ_sigma():

    s0=5
    mu=4
    theta=2
    dt=0.25
    sigma=0.32
    T=4

    #################################
    # test time varying sigma
    #################################
    sigmas = np.ones(16) * sigma

    df1 = rt.simOUJ(s0=s0, mu=mu, theta=theta, sigma=sigma, T=T, dt=dt, sims=2, seed=12345) 
    df2 = rt.simOUJ(s0=s0, mu=mu, theta=theta, sigma=sigmas, T=T, dt=dt, sims=2, seed=12345) 
    assert np.allclose(df1, df2), "Py time varying sigma test failed"

    # test time varying and 2D sigma

    sigma = np.ones((252, 1000)) * 0.1
    sigma[100:,:] = 0.5

    df = rt.simOUJ(sigma=sigma, theta=20, T=1, dt=1/252, sims=1000, seed=12345)

    assert df.iloc[152,:].std()/df.iloc[25,:].std() > 3, "Time varying sigma test failed"



if __name__ == "__main__":
    test_simOU_eps()


