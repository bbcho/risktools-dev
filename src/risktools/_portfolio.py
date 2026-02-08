# Portfolio Optimization and Risk Functions

from typing import Union, Optional

import numpy as _np
import pandas as _pd
import scipy.optimize as _opt
import scipy.stats as _stats

__all__ = [
    "portfolio_optimize",
    "portfolio_var",
    "portfolio_performance",
]

# Mapping from pandas frequency strings to annualization factors
_FREQ_SCALE = {
    "D": 252, "B": 252,
    "W": 52,
    "M": 12, "ME": 12, "MS": 12,
    "Q": 4, "QE": 4, "QS": 4,
    "Y": 1, "YE": 1, "YS": 1, "A": 1, "AS": 1,
}


def _infer_scale(returns: _pd.DataFrame) -> int:
    r"""
    Infer annualization factor from a DataFrame's DatetimeIndex frequency.

    Attempts to determine the number of periods per year by inspecting the
    frequency attribute of the DataFrame's index. Falls back to 252 (daily)
    if no frequency can be inferred.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with a DatetimeIndex whose ``freq`` attribute is set.

    Returns
    -------
    int
        Annualization factor (e.g. 252 for daily, 12 for monthly).

    Notes
    -----
    The function first checks ``returns.index.freq`` using ``freqstr`` and
    ``name`` attributes, then attempts prefix matching for composite
    frequency strings like ``"W-FRI"``.  If no frequency is found on the
    index, :func:`pandas.infer_freq` is tried as a last resort before
    defaulting to 252.
    """
    freq = None
    if isinstance(returns.index, _pd.DatetimeIndex):
        freq = returns.index.freq

    if freq is not None:
        freq_str = getattr(freq, "freqstr", None) or getattr(freq, "name", None) or str(freq)
        if freq_str in _FREQ_SCALE:
            return _FREQ_SCALE[freq_str]
        # Try prefix match for frequencies like 'W-FRI'
        if freq_str and freq_str[0] in _FREQ_SCALE:
            return _FREQ_SCALE[freq_str[0]]

    # Try pandas infer_freq as fallback
    if isinstance(returns.index, _pd.DatetimeIndex) and len(returns.index) >= 3:
        inferred = _pd.infer_freq(returns.index)
        if inferred is not None:
            if inferred in _FREQ_SCALE:
                return _FREQ_SCALE[inferred]
            if inferred[0] in _FREQ_SCALE:
                return _FREQ_SCALE[inferred[0]]

    # Default to daily
    return 252


def portfolio_optimize(
    returns: _pd.DataFrame,
    method: str = "min_variance",
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    constraints: Optional[dict] = None,
) -> dict:
    r"""
    Mean-variance portfolio optimization.

    Compute optimal portfolio weights using one of several classical
    portfolio construction methods: minimum variance, maximum Sharpe ratio
    (tangency portfolio), risk parity, or minimum variance subject to a
    target return.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset return series where each column represents one
        asset.  Must contain at least two columns and have a DatetimeIndex
        for automatic annualization; otherwise a daily frequency (scale=252)
        is assumed.
    method : str, optional
        Optimization method, one of:

        - ``"min_variance"`` : Minimum variance portfolio.
        - ``"max_sharpe"`` : Maximum Sharpe ratio (tangency) portfolio.
        - ``"risk_parity"`` : Risk parity (equal risk contribution).
        - ``"target_return"`` : Minimum variance for a specified target return.

        Default is ``"min_variance"``.
    target_return : float, optional
        Required when ``method="target_return"``.  The desired annualized
        portfolio return.
    risk_free_rate : float, optional
        Annualized risk-free rate used for Sharpe ratio calculation.
        Default is ``0.0``.
    constraints : dict, optional
        Portfolio constraints.  Supported keys:

        - ``"long_only"`` : bool, default ``True``.  If True, all weights
          must be non-negative.
        - ``"max_weight"`` : float, default ``1.0``.  Upper bound on each
          individual asset weight.
        - ``"min_weight"`` : float, default ``0.0``.  Lower bound on each
          individual asset weight (overridden to 0 when ``long_only=True``
          and ``min_weight < 0``).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"weights"`` : pd.Series indexed by asset names from the column
          labels of *returns*.
        - ``"expected_return"`` : float, annualized expected return.
        - ``"volatility"`` : float, annualized portfolio volatility.
        - ``"sharpe_ratio"`` : float, annualized Sharpe ratio.

    Notes
    -----
    **Minimum Variance**

    .. math:: \min_{\mathbf{w}} \; \mathbf{w}^\top \Sigma \mathbf{w}
              \quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1

    **Maximum Sharpe Ratio (Tangency Portfolio)**

    .. math:: \max_{\mathbf{w}} \;
              \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}
                   {\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}
              \quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1

    **Risk Parity**

    Each asset's marginal risk contribution is equalized:

    .. math:: \min_{\mathbf{w}} \sum_{i=1}^{n}
              \left(
              \frac{w_i \, (\Sigma \mathbf{w})_i}
                   {\mathbf{w}^\top \Sigma \mathbf{w}}
              - \frac{1}{n}
              \right)^2
              \quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1

    **Target Return**

    .. math:: \min_{\mathbf{w}} \; \mathbf{w}^\top \Sigma \mathbf{w}
              \quad \text{s.t.} \quad
              \mathbf{w}^\top \boldsymbol{\mu} = \mu^*,\;
              \mathbf{1}^\top \mathbf{w} = 1

    Annualization uses :math:`\text{scale}` periods per year (inferred from
    the index frequency, defaulting to 252):

    .. math::

        \mu_{\text{ann}} = \mathbf{w}^\top \boldsymbol{\mu} \cdot
        \text{scale}

        \sigma_{\text{ann}} = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}
        \cdot \text{scale}}

    Optimization is performed using ``scipy.optimize.minimize`` with the
    SLSQP method.

    Raises
    ------
    ValueError
        If *returns* has fewer than two columns, *method* is unrecognized,
        or ``method="target_return"`` is used without specifying
        *target_return*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> returns = pd.DataFrame(
    ...     np.random.randn(500, 3) * 0.01,
    ...     index=dates,
    ...     columns=["A", "B", "C"],
    ... )
    >>> result = portfolio_optimize(returns, method="min_variance")
    >>> result["weights"].sum()  # doctest: +SKIP
    1.0
    """
    _valid_methods = {"min_variance", "max_sharpe", "risk_parity", "target_return"}
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {_valid_methods!r}, got {method!r}"
        )

    if not isinstance(returns, _pd.DataFrame) or returns.shape[1] < 1:
        raise ValueError("returns must be a pandas DataFrame with at least one column")

    # Drop rows with any NaN
    returns = returns.dropna()
    n_assets = returns.shape[1]
    asset_names = returns.columns.tolist()

    # Single asset short-circuit
    if n_assets == 1:
        scale = _infer_scale(returns)
        mu_ann = float(returns.iloc[:, 0].mean() * scale)
        vol_ann = float(returns.iloc[:, 0].std(ddof=1) * _np.sqrt(scale))
        sr = (mu_ann - risk_free_rate) / vol_ann if vol_ann > 0 else 0.0
        return {
            "weights": _pd.Series([1.0], index=asset_names),
            "expected_return": mu_ann,
            "volatility": vol_ann,
            "sharpe_ratio": sr,
        }

    # Parse constraints
    if constraints is None:
        constraints = {}
    long_only = constraints.get("long_only", True)
    max_weight = constraints.get("max_weight", 1.0)
    min_weight = constraints.get("min_weight", 0.0)

    if long_only and min_weight < 0:
        min_weight = 0.0

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Moments
    mu = returns.mean().values  # (n,) per-period mean
    cov = returns.cov().values  # (n, n) per-period covariance
    scale = _infer_scale(returns)

    # Initial guess: equal weight
    w0 = _np.full(n_assets, 1.0 / n_assets)

    # Fully-invested constraint
    sum_to_one = {"type": "eq", "fun": lambda w: _np.sum(w) - 1.0}

    def _portfolio_variance(w):
        return w @ cov @ w

    def _portfolio_return(w):
        return w @ mu

    if method == "min_variance":
        result = _opt.minimize(
            _portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_to_one],
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    elif method == "max_sharpe":
        def _neg_sharpe(w):
            port_ret = w @ mu * scale
            port_vol = _np.sqrt(w @ cov @ w * scale)
            if port_vol < 1e-16:
                return 0.0
            return -(port_ret - risk_free_rate) / port_vol

        result = _opt.minimize(
            _neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_to_one],
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    elif method == "risk_parity":
        def _risk_parity_objective(w):
            port_var = w @ cov @ w
            if port_var < 1e-16:
                return 1e10
            marginal_contrib = cov @ w  # (n,)
            risk_contrib = w * marginal_contrib  # (n,)
            total_risk = port_var
            target_contrib = 1.0 / n_assets
            return _np.sum(
                (risk_contrib / total_risk - target_contrib) ** 2
            )

        result = _opt.minimize(
            _risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_to_one],
            options={"ftol": 1e-15, "maxiter": 2000},
        )

    elif method == "target_return":
        if target_return is None:
            raise ValueError(
                "target_return must be specified when method='target_return'"
            )
        # target_return is annualized; convert to per-period
        target_per_period = target_return / scale

        target_constraint = {
            "type": "eq",
            "fun": lambda w: w @ mu - target_per_period,
        }

        result = _opt.minimize(
            _portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_to_one, target_constraint],
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    if not result.success:
        import warnings
        warnings.warn(
            f"Optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    weights = result.x

    # Normalize weights to sum to exactly 1 (numerical cleanup)
    weights = weights / _np.sum(weights)

    # Annualized metrics
    ann_return = float(weights @ mu * scale)
    ann_vol = float(_np.sqrt(weights @ cov @ weights * scale))
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-16 else 0.0

    return {
        "weights": _pd.Series(weights, index=asset_names),
        "expected_return": ann_return,
        "volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
    }


def portfolio_var(
    returns: _pd.DataFrame,
    weights: Union[_np.ndarray, _pd.Series, list],
    alpha: float = 0.05,
    method: str = "parametric",
) -> float:
    r"""
    Portfolio Value at Risk (VaR).

    Estimate the Value at Risk of a portfolio given constituent asset
    returns and portfolio weights.  Three estimation methods are supported:
    parametric (variance-covariance), historical simulation, and Monte
    Carlo simulation.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset return series.  Each column is one asset.
    weights : array-like or pd.Series
        Portfolio weights.  Must have the same length as the number of
        columns in *returns*.
    alpha : float, optional
        Significance level.  The VaR is computed at the
        :math:`(1-\alpha)` confidence level.  Default is ``0.05`` (95 %
        confidence).
    method : str, optional
        Estimation method, one of:

        - ``"parametric"`` : Gaussian VaR using portfolio mean and standard
          deviation.
        - ``"historical"`` : Empirical quantile of the realized portfolio
          return series.
        - ``"monte_carlo"`` : Quantile from simulated returns drawn from a
          fitted multivariate normal distribution (10 000 draws).

        Default is ``"parametric"``.

    Returns
    -------
    float
        Value at Risk expressed as a **positive** number representing the
        loss at the given confidence level for a single period.

    Notes
    -----
    **Parametric (variance-covariance) VaR**

    Under the assumption that portfolio returns are normally distributed:

    .. math:: \text{VaR}_\alpha = -\left(\mu_p +
              z_\alpha \, \sigma_p\right)

    where :math:`\mu_p = \mathbf{w}^\top \boldsymbol{\mu}`,
    :math:`\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}`, and
    :math:`z_\alpha = \Phi^{-1}(\alpha)`.

    **Historical VaR**

    .. math:: \text{VaR}_\alpha = -Q_\alpha(R_p)

    where :math:`Q_\alpha` denotes the :math:`\alpha`-quantile of the
    realized portfolio return series
    :math:`R_p = \mathbf{R} \mathbf{w}`.

    **Monte Carlo VaR**

    Simulate :math:`N=10\,000` return vectors from
    :math:`\mathcal{N}(\boldsymbol{\mu}, \Sigma)`, compute portfolio
    returns, then take the empirical :math:`\alpha`-quantile.

    Raises
    ------
    ValueError
        If *method* is not recognized, or *weights* length does not match
        the number of columns in *returns*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> rets = pd.DataFrame(
    ...     np.random.randn(500, 3) * 0.01,
    ...     index=dates, columns=["A", "B", "C"],
    ... )
    >>> w = np.array([0.4, 0.4, 0.2])
    >>> portfolio_var(rets, w, alpha=0.05, method="historical")  # doctest: +SKIP
    0.012...
    """
    _valid_methods = {"parametric", "historical", "monte_carlo"}
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {_valid_methods!r}, got {method!r}"
        )

    returns = returns.dropna()
    w = _np.asarray(weights, dtype=float).ravel()

    if w.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match number of assets "
            f"({returns.shape[1]})"
        )

    if method == "parametric":
        port_returns = returns.values @ w
        mu_p = float(_np.mean(port_returns))
        sigma_p = float(_np.std(port_returns, ddof=1))
        z = _stats.norm.ppf(alpha)
        var = -(mu_p + z * sigma_p)

    elif method == "historical":
        port_returns = returns.values @ w
        var = -float(_np.percentile(port_returns, alpha * 100))

    elif method == "monte_carlo":
        mu = returns.mean().values
        cov = returns.cov().values
        n_sims = 10_000
        rng = _np.random.default_rng(42)
        sim_returns = rng.multivariate_normal(mu, cov, size=n_sims)
        port_returns = sim_returns @ w
        var = -float(_np.percentile(port_returns, alpha * 100))

    return max(var, 0.0)


def portfolio_performance(
    returns: _pd.DataFrame,
    weights: Union[_np.ndarray, _pd.Series, list],
    risk_free_rate: float = 0.0,
) -> dict:
    r"""
    Compute key portfolio performance metrics.

    Given a DataFrame of asset returns and a weight vector, compute
    annualized return, volatility, Sharpe ratio, maximum drawdown, Calmar
    ratio, and Sortino ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset return series.  Each column is one asset.
    weights : array-like or pd.Series
        Portfolio weights.  Must have the same length as the number of
        columns in *returns*.
    risk_free_rate : float, optional
        Annualized risk-free rate used for ratio calculations.  Default is
        ``0.0``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"annualized_return"`` : float
        - ``"annualized_volatility"`` : float
        - ``"sharpe_ratio"`` : float
        - ``"max_drawdown"`` : float (positive number)
        - ``"calmar_ratio"`` : float
        - ``"sortino_ratio"`` : float

    Notes
    -----
    **Annualized Return** (geometric)

    .. math:: R_{\text{ann}} =
              \left(\prod_{t=1}^{T}(1 + R_{p,t})\right)^{
              \frac{\text{scale}}{T}} - 1

    **Annualized Volatility**

    .. math:: \sigma_{\text{ann}} = \sigma(R_p) \cdot \sqrt{\text{scale}}

    **Sharpe Ratio**

    .. math:: S = \frac{R_{\text{ann}} - r_f}{\sigma_{\text{ann}}}

    **Maximum Drawdown**

    .. math:: \text{MDD} = \max_{t \in [0,T]}
              \left(
              \frac{\max_{s \in [0,t]} V_s - V_t}{\max_{s \in [0,t]} V_s}
              \right)

    where :math:`V_t` is the cumulative wealth index
    :math:`\prod_{s=1}^{t}(1 + R_{p,s})`.

    **Calmar Ratio**

    .. math:: C = \frac{R_{\text{ann}}}{\text{MDD}}

    **Sortino Ratio**

    Uses downside deviation (target = 0) as the denominator:

    .. math:: \text{Sortino} =
              \frac{R_{\text{ann}} - r_f}{\sigma_d \cdot \sqrt{\text{scale}}}

    where

    .. math:: \sigma_d =
              \sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(R_{p,t}, 0)^2}

    Raises
    ------
    ValueError
        If *weights* length does not match the number of columns in
        *returns*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> rets = pd.DataFrame(
    ...     np.random.randn(500, 3) * 0.01 + 0.0003,
    ...     index=dates, columns=["A", "B", "C"],
    ... )
    >>> w = np.array([0.5, 0.3, 0.2])
    >>> perf = portfolio_performance(rets, w)
    >>> sorted(perf.keys())  # doctest: +NORMALIZE_WHITESPACE
    ['annualized_return', 'annualized_volatility', 'calmar_ratio',
     'max_drawdown', 'sharpe_ratio', 'sortino_ratio']
    """
    returns = returns.dropna()
    w = _np.asarray(weights, dtype=float).ravel()

    if w.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match number of assets "
            f"({returns.shape[1]})"
        )

    scale = _infer_scale(returns)

    # Portfolio return series
    port_returns = _pd.Series(
        returns.values @ w,
        index=returns.index,
    )

    n = len(port_returns)

    # Annualized return (geometric)
    cumulative = (1 + port_returns).prod()
    ann_return = float(cumulative ** (scale / n) - 1)

    # Annualized volatility
    ann_vol = float(port_returns.std(ddof=1) * _np.sqrt(scale))

    # Sharpe ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-16 else 0.0

    # Maximum drawdown
    wealth_index = (1 + port_returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (running_max - wealth_index) / running_max
    max_dd = float(drawdown.max())

    # Calmar ratio
    calmar = ann_return / max_dd if max_dd > 1e-16 else 0.0

    # Sortino ratio
    downside = port_returns.clip(upper=0.0)
    downside_dev = float(_np.sqrt((downside ** 2).mean()))
    ann_downside_dev = downside_dev * _np.sqrt(scale)
    sortino = (
        (ann_return - risk_free_rate) / ann_downside_dev
        if ann_downside_dev > 1e-16
        else 0.0
    )

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "calmar_ratio": float(calmar),
        "sortino_ratio": float(sortino),
    }
