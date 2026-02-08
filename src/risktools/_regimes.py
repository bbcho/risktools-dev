# Regime Detection Module

from typing import Optional, Union, List

import numpy as _np
import pandas as _pd
from scipy.stats import norm as _norm
from scipy.special import logsumexp as _logsumexp

__all__ = [
    "detect_regimes",
    "structural_break",
    "regime_summary",
]


def _log_gaussian_pdf(x, mu, var):
    r"""
    Compute log of Gaussian PDF for numerical stability.

    .. math:: \log \mathcal{N}(x \mid \mu, \sigma^2)
        = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}

    Parameters
    ----------
    x : np.ndarray
        Observations, shape (T,).
    mu : float
        Mean of the Gaussian.
    var : float
        Variance of the Gaussian (must be > 0).

    Returns
    -------
    np.ndarray
        Log-probabilities, shape (T,).
    """
    return -0.5 * _np.log(2.0 * _np.pi * var) - 0.5 * (x - mu) ** 2 / var


def _forward(log_obs, log_A, log_pi):
    r"""
    Forward algorithm in log-space.

    Computes :math:`\log \alpha_t(k)` where
    :math:`\alpha_t(k) = P(O_1 \ldots O_t, S_t = k)`.

    Parameters
    ----------
    log_obs : np.ndarray
        Log emission probabilities, shape (T, K).
    log_A : np.ndarray
        Log transition matrix, shape (K, K).
    log_pi : np.ndarray
        Log initial state distribution, shape (K,).

    Returns
    -------
    log_alpha : np.ndarray
        Log forward variables, shape (T, K).
    """
    T, K = log_obs.shape
    log_alpha = _np.full((T, K), -_np.inf)

    # t = 0
    log_alpha[0] = log_pi + log_obs[0]

    for t in range(1, T):
        for k in range(K):
            # log_alpha[t, k] = log( sum_j alpha[t-1,j] * A[j,k] ) + log_obs[t,k]
            log_alpha[t, k] = (
                _logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_obs[t, k]
            )

    return log_alpha


def _backward(log_obs, log_A):
    r"""
    Backward algorithm in log-space.

    Computes :math:`\log \beta_t(k)` where
    :math:`\beta_t(k) = P(O_{t+1} \ldots O_T \mid S_t = k)`.

    Parameters
    ----------
    log_obs : np.ndarray
        Log emission probabilities, shape (T, K).
    log_A : np.ndarray
        Log transition matrix, shape (K, K).

    Returns
    -------
    log_beta : np.ndarray
        Log backward variables, shape (T, K).
    """
    T, K = log_obs.shape
    log_beta = _np.full((T, K), -_np.inf)

    # t = T-1 (last time step)
    log_beta[T - 1] = 0.0  # log(1) = 0

    for t in range(T - 2, -1, -1):
        for k in range(K):
            # log_beta[t, k] = log( sum_j A[k,j] * obs[t+1,j] * beta[t+1,j] )
            log_beta[t, k] = _logsumexp(
                log_A[k, :] + log_obs[t + 1] + log_beta[t + 1]
            )

    return log_beta


def _viterbi(log_obs, log_A, log_pi):
    r"""
    Viterbi algorithm for most likely state sequence.

    Finds :math:`\arg\max_{S_1 \ldots S_T} P(S_1 \ldots S_T \mid O_1 \ldots O_T)`.

    Parameters
    ----------
    log_obs : np.ndarray
        Log emission probabilities, shape (T, K).
    log_A : np.ndarray
        Log transition matrix, shape (K, K).
    log_pi : np.ndarray
        Log initial state distribution, shape (K,).

    Returns
    -------
    states : np.ndarray
        Most likely state sequence, shape (T,), dtype int.
    """
    T, K = log_obs.shape
    # delta[t, k] = max log-prob of path ending in state k at time t
    delta = _np.full((T, K), -_np.inf)
    psi = _np.zeros((T, K), dtype=int)

    # Initialization
    delta[0] = log_pi + log_obs[0]

    # Recursion
    for t in range(1, T):
        for k in range(K):
            candidates = delta[t - 1] + log_A[:, k]
            psi[t, k] = _np.argmax(candidates)
            delta[t, k] = candidates[psi[t, k]] + log_obs[t, k]

    # Backtracking
    states = _np.zeros(T, dtype=int)
    states[T - 1] = _np.argmax(delta[T - 1])

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def detect_regimes(
    returns: _pd.Series,
    n_regimes: int = 2,
    method: str = "gaussian_hmm",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> dict:
    r"""
    Detect market regimes using a Gaussian Hidden Markov Model.

    Fits a Gaussian HMM to the return series using the Baum-Welch
    (Expectation-Maximization) algorithm, then decodes the most likely
    state sequence via the Viterbi algorithm.

    The model assumes that observations are generated from one of
    :math:`K` hidden states, each with its own Gaussian emission
    distribution:

    .. math:: O_t \mid S_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)

    State transitions follow a first-order Markov chain with transition
    matrix :math:`A` where :math:`A_{ij} = P(S_{t+1} = j \mid S_t = i)`.

    Parameters
    ----------
    returns : pd.Series
        Series of returns with a datetime index.
    n_regimes : int, optional
        Number of hidden states (regimes), by default 2 (e.g., bull/bear).
    method : str, optional
        Estimation method. Currently only ``"gaussian_hmm"`` is supported.
    max_iter : int, optional
        Maximum number of EM iterations, by default 100.
    tol : float, optional
        Convergence tolerance on the change in log-likelihood between
        successive iterations, by default 1e-4.
    seed : int or None, optional
        Random seed for reproducible initialization, by default None.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"states"`` : pd.Series of most likely state assignments
          (Viterbi), same index as *returns*. States are sorted so that
          state 0 has the lowest mean.
        - ``"means"`` : np.ndarray of shape (K,), regime means sorted
          ascending.
        - ``"variances"`` : np.ndarray of shape (K,), regime variances
          (ordered to match means).
        - ``"transition_matrix"`` : np.ndarray of shape (K, K), the
          estimated transition probability matrix.
        - ``"log_likelihood"`` : float, final log-likelihood.
        - ``"n_iter"`` : int, number of EM iterations until convergence.

    Notes
    -----
    The implementation uses log-space arithmetic throughout the forward,
    backward, and Viterbi passes to prevent numerical underflow. The
    log-sum-exp trick is:

    .. math:: \log \sum_i e^{x_i}
        = \max(x) + \log \sum_i e^{x_i - \max(x)}

    A variance floor of :math:`10^{-10}` is enforced to prevent degenerate
    states during EM.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500)
    >>> rng = np.random.default_rng(42)
    >>> returns = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
    >>> result = detect_regimes(returns, n_regimes=2, seed=42)
    >>> result["means"].shape
    (2,)
    """
    if method != "gaussian_hmm":
        raise ValueError(
            f"Unknown method '{method}'. Only 'gaussian_hmm' is supported."
        )

    obs = _np.asarray(returns, dtype=float)
    T = len(obs)
    K = n_regimes
    rng = _np.random.default_rng(seed)

    _VAR_FLOOR = 1e-10

    # ------------------------------------------------------------------
    # 1. Initialization
    # ------------------------------------------------------------------
    # Transition matrix: slightly favor self-transitions
    A = _np.full((K, K), 0.1 / (K - 1)) if K > 1 else _np.ones((1, 1))
    if K > 1:
        _np.fill_diagonal(A, 0.9)
    # Normalize rows
    A = A / A.sum(axis=1, keepdims=True)

    # Initial state distribution: uniform
    pi = _np.ones(K) / K

    # Means: spread across quantiles of the data
    quantiles = _np.linspace(0, 1, K + 2)[1:-1]  # exclude 0 and 1
    means = _np.quantile(obs, quantiles) + rng.normal(0, 1e-6, K)

    # Variances: initialize from data variance, slightly perturbed
    data_var = _np.var(obs)
    variances = _np.full(K, max(data_var, _VAR_FLOOR)) * (
        1.0 + rng.normal(0, 0.1, K)
    )
    variances = _np.maximum(variances, _VAR_FLOOR)

    # ------------------------------------------------------------------
    # 2. EM (Baum-Welch) iterations
    # ------------------------------------------------------------------
    prev_ll = -_np.inf

    for iteration in range(1, max_iter + 1):
        # Log parameters
        log_A = _np.log(A)
        log_pi = _np.log(pi)

        # Emission log-probabilities: shape (T, K)
        log_obs = _np.column_stack(
            [_log_gaussian_pdf(obs, means[k], variances[k]) for k in range(K)]
        )

        # E-step: forward-backward
        log_alpha = _forward(log_obs, log_A, log_pi)
        log_beta = _backward(log_obs, log_A)

        # Log-likelihood
        ll = _logsumexp(log_alpha[T - 1])

        # Check convergence
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # Posterior state probabilities gamma_t(k) = P(S_t=k | O)
        log_gamma = log_alpha + log_beta
        # Normalize across states for each time step
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = _np.exp(log_gamma)

        # Transition posteriors xi_t(i,j) for t=0..T-2
        # xi_t(i,j) = alpha_t(i) * A_ij * b_j(O_{t+1}) * beta_{t+1}(j) / P(O)
        log_xi = _np.full((T - 1, K, K), -_np.inf)
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    log_xi[t, i, j] = (
                        log_alpha[t, i]
                        + log_A[i, j]
                        + log_obs[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            # Normalize
            log_xi[t] = log_xi[t] - _logsumexp(log_xi[t].ravel())

        xi = _np.exp(log_xi)

        # M-step
        # Transition matrix
        for i in range(K):
            denom = gamma[:T - 1, i].sum()
            if denom > 0:
                for j in range(K):
                    A[i, j] = xi[:, i, j].sum() / denom
            else:
                A[i, :] = 1.0 / K
        # Normalize rows to handle numerical drift
        A = A / A.sum(axis=1, keepdims=True)

        # Initial distribution
        pi = gamma[0]
        pi = pi / pi.sum()

        # Means and variances
        for k in range(K):
            gamma_k = gamma[:, k]
            gamma_sum = gamma_k.sum()
            if gamma_sum > 0:
                means[k] = (gamma_k * obs).sum() / gamma_sum
                variances[k] = (gamma_k * (obs - means[k]) ** 2).sum() / gamma_sum
                variances[k] = max(variances[k], _VAR_FLOOR)
            # If gamma_sum == 0, keep previous parameters (state unused)

    n_iter = iteration

    # ------------------------------------------------------------------
    # 3. Sort regimes by mean (state 0 = lowest mean)
    # ------------------------------------------------------------------
    order = _np.argsort(means)
    means = means[order]
    variances = variances[order]
    A = A[_np.ix_(order, order)]
    pi = pi[order]

    # Build mapping from old labels to new sorted labels
    label_map = _np.zeros(K, dtype=int)
    for new_idx, old_idx in enumerate(order):
        label_map[old_idx] = new_idx

    # ------------------------------------------------------------------
    # 4. Viterbi decoding with sorted parameters
    # ------------------------------------------------------------------
    log_A = _np.log(A)
    log_pi = _np.log(pi)
    log_obs = _np.column_stack(
        [_log_gaussian_pdf(obs, means[k], variances[k]) for k in range(K)]
    )
    states_arr = _viterbi(log_obs, log_A, log_pi)

    states = _pd.Series(states_arr, index=returns.index, name="regime")

    return {
        "states": states,
        "means": means,
        "variances": variances,
        "transition_matrix": A,
        "log_likelihood": ll,
        "n_iter": n_iter,
    }


def structural_break(
    series: _pd.Series,
    method: str = "cusum",
    threshold: Optional[float] = None,
    drift: float = 0.0,
) -> dict:
    r"""
    Detect structural breaks using CUSUM or CUSUM-of-squares.

    The **CUSUM** (Cumulative Sum) method detects shifts in the mean of a
    process by accumulating standardized deviations. The **CUSUM of squares**
    variant detects changes in variance.

    CUSUM algorithm (standardized):

    .. math::

        z_t = \frac{x_t - \mu}{\sigma}

        S_t^{+} = \max(0,\; S_{t-1}^{+} + z_t - k)

        S_t^{-} = \max(0,\; S_{t-1}^{-} - z_t - k)

    where :math:`\mu` and :math:`\sigma` are the sample mean and standard
    deviation, and :math:`k` is the drift (allowance) parameter.  A
    structural break is signalled when :math:`S_t^{+} > h` or
    :math:`S_t^{-} > h`.

    CUSUM of squares:

    .. math::

        W_t = \frac{\sum_{i=1}^{t} e_i^2}{\sum_{i=1}^{n} e_i^2}

        D_t = W_t - \frac{t}{n}

    where :math:`e_i = x_i - \bar{x}`.  A break is signalled when
    :math:`\max |D_t|` exceeds the critical value.

    Parameters
    ----------
    series : pd.Series
        Time series of values (prices, returns, or any numeric series).
    method : str, optional
        Detection method, one of:

        - ``"cusum"`` : standard CUSUM test for mean shifts (default).
        - ``"cusum_sq"`` : CUSUM of squares for variance changes.
    threshold : float or None, optional
        Detection threshold *h*. If ``None``:

        - For ``"cusum"``: :math:`h = 4` (in units of standard deviations,
          since the series is standardized).
        - For ``"cusum_sq"``: :math:`h = 1.358 / \sqrt{n} + 0.12`
          (approximate 5 %% critical value from Ploberger & Kramer).
    drift : float, optional
        Allowance parameter *k* (default 0.0). A positive drift makes the
        detector less sensitive to small shifts.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"cusum_pos"`` : pd.Series, positive CUSUM statistic
          :math:`S_t^{+}` (or :math:`D_t^{+}` for cusum_sq).
        - ``"cusum_neg"`` : pd.Series, negative CUSUM statistic as
          positive values (i.e., :math:`|S_t^{-}|`).
        - ``"breaks"`` : list of index labels where breaks are detected
          (threshold exceeded by either statistic).
        - ``"threshold"`` : float, the threshold used.
        - ``"statistics"`` : pd.DataFrame with columns
          ``["cusum_pos", "cusum_neg"]``, same index as *series*.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range("2020-01-01", periods=200, freq="B")
    >>> rng = np.random.default_rng(0)
    >>> vals = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    >>> s = pd.Series(vals, index=idx)
    >>> result = structural_break(s, method="cusum")
    >>> len(result["breaks"]) > 0
    True
    """
    x = _np.asarray(series, dtype=float)
    n = len(x)

    if method == "cusum":
        mu = _np.mean(x)
        sigma = _np.std(x, ddof=1)
        if sigma < 1e-15:
            sigma = 1.0  # prevent division by zero for constant series

        z = (x - mu) / sigma
        k = drift

        if threshold is None:
            threshold = 4.0

        s_pos = _np.zeros(n)
        s_neg = _np.zeros(n)

        for t in range(n):
            if t == 0:
                s_pos[t] = max(0.0, z[t] - k)
                s_neg[t] = max(0.0, -z[t] - k)
            else:
                s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - k)
                s_neg[t] = max(0.0, s_neg[t - 1] - z[t] - k)

        cusum_pos = _pd.Series(s_pos, index=series.index, name="cusum_pos")
        cusum_neg = _pd.Series(s_neg, index=series.index, name="cusum_neg")

        # Detect break points
        break_mask = (s_pos > threshold) | (s_neg > threshold)
        breaks = list(series.index[break_mask])

    elif method == "cusum_sq":
        mu = _np.mean(x)
        residuals = x - mu
        e_sq = residuals ** 2
        total_sq = e_sq.sum()

        if total_sq < 1e-15:
            # Constant series, no variance to detect
            cusum_pos = _pd.Series(
                _np.zeros(n), index=series.index, name="cusum_pos"
            )
            cusum_neg = _pd.Series(
                _np.zeros(n), index=series.index, name="cusum_neg"
            )
            return {
                "cusum_pos": cusum_pos,
                "cusum_neg": cusum_neg,
                "breaks": [],
                "threshold": 0.0,
                "statistics": _pd.DataFrame(
                    {"cusum_pos": cusum_pos, "cusum_neg": cusum_neg}
                ),
            }

        # W_t = cumulative sum of squared residuals / total
        W = _np.cumsum(e_sq) / total_sq
        # D_t = W_t - t/n
        t_over_n = _np.arange(1, n + 1) / n
        D = W - t_over_n

        # Split D into positive and negative parts (as positive values)
        s_pos = _np.maximum(D, 0.0)
        s_neg = _np.maximum(-D, 0.0)

        cusum_pos = _pd.Series(s_pos, index=series.index, name="cusum_pos")
        cusum_neg = _pd.Series(s_neg, index=series.index, name="cusum_neg")

        if threshold is None:
            # Approximate 5% critical value (Ploberger & Kramer)
            threshold = 1.358 / _np.sqrt(n) + 0.12

        break_mask = (_np.abs(D) > threshold)
        breaks = list(series.index[break_mask])

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'cusum' or 'cusum_sq'."
        )

    statistics = _pd.DataFrame(
        {"cusum_pos": cusum_pos, "cusum_neg": cusum_neg}
    )

    return {
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "breaks": breaks,
        "threshold": threshold,
        "statistics": statistics,
    }


def regime_summary(
    returns: _pd.Series,
    states: _pd.Series,
) -> _pd.DataFrame:
    r"""
    Summarize statistics for each detected regime.

    Computes annualized performance metrics and distributional statistics
    for each regime identified by ``detect_regimes``.

    Annualization assumes 252 trading days per year:

    .. math::

        \text{annualized mean} = \bar{r}_k \times 252

        \text{annualized vol} = \sigma_k \times \sqrt{252}

        \text{Sharpe} = \frac{\bar{r}_k \times 252}{\sigma_k \times \sqrt{252}}

    Parameters
    ----------
    returns : pd.Series
        Series of returns.
    states : pd.Series
        Series of regime labels with the same index as *returns*
        (e.g., from ``detect_regimes(...)["states"]``).

    Returns
    -------
    pd.DataFrame
        DataFrame with index = regime labels (sorted), columns:

        - ``"mean"`` : annualized mean return.
        - ``"volatility"`` : annualized volatility.
        - ``"sharpe"`` : annualized Sharpe ratio.
        - ``"skewness"`` : skewness of daily returns in regime.
        - ``"kurtosis"`` : excess kurtosis of daily returns in regime.
        - ``"count"`` : number of observations in regime.
        - ``"pct"`` : percentage of total observations in regime.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500)
    >>> rng = np.random.default_rng(42)
    >>> returns = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
    >>> result = detect_regimes(returns, n_regimes=2, seed=42)
    >>> summary = regime_summary(returns, result["states"])
    >>> list(summary.columns)
    ['mean', 'volatility', 'sharpe', 'skewness', 'kurtosis', 'count', 'pct']
    """
    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    labels = sorted(states.unique())
    total = len(returns)
    rows = []

    for label in labels:
        mask = states == label
        r = returns[mask]
        n = len(r)

        daily_mean = r.mean()
        daily_std = r.std(ddof=1)

        ann_mean = daily_mean * 252
        ann_vol = daily_std * _np.sqrt(252)
        sharpe = ann_mean / ann_vol if ann_vol > 0 else _np.nan

        sk = float(_skew(r, bias=False)) if n >= 3 else _np.nan
        kt = float(_kurtosis(r, fisher=True, bias=False)) if n >= 4 else _np.nan

        rows.append(
            {
                "mean": ann_mean,
                "volatility": ann_vol,
                "sharpe": sharpe,
                "skewness": sk,
                "kurtosis": kt,
                "count": n,
                "pct": 100.0 * n / total,
            }
        )

    df = _pd.DataFrame(rows, index=labels)
    df.index.name = "regime"

    return df
