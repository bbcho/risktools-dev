# multivariate simulations

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import plotly.graph_objects as _go


def simGBM_MV(s0, r, sigma, cor, T, dt, sims, eps=None):
    """
    Simulate geometric Brownian motion
    """
    N = int(T / dt)
    # S = _np.zeros((N+1, sims))

    if eps is None:
        cov = sigma @ cor @ sigma
        eps = _np.random.multivariate_normal(_np.zeros(len(r)), cov, size=(N + 1, sims))

    # S _np.exp((mu-0.5*sigma**2)*dt+sigma*_np.sqrt(dt)*_np.random.normal(size=M))

    return eps


if __name__ == "__main__":
    simGBM_MV()
