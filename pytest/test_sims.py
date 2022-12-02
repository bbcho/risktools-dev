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
