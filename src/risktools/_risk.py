# Core Risk Measurement Functions
# VaR, CVaR (Expected Shortfall), CFaR (Cash Flow at Risk),
# NPVaR (NPV at Risk), rolling risk measures, portfolio VaR

from typing import Union, Optional

import numpy as _np
import pandas as _pd
from scipy.stats import norm as _norm

__all__ = [
    "value_at_risk",
    "cvar",
    "cfar",
    "npv_at_risk",
    "rolling_var",
    "rolling_cvar",
    "parametric_var_portfolio",
]


def value_at_risk(
    returns: _pd.Series,
    confidence: float = 0.95,
    method: str = "parametric",
    n_sims: int = 10000,
    random_state: Optional[int] = None,
) -> float:
    r"""
    Compute Value at Risk (VaR) for a return series.

    VaR is the maximum expected loss over a single period at a given
    confidence level.  By convention the returned value is **negative**
    (representing a loss in the profit-and-loss sense).

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic asset returns (arithmetic or log).
    confidence : float, optional
        Confidence level (e.g. 0.95 for 95 %).  Default is ``0.95``.
    method : str, optional
        Estimation method, one of:

        - ``"parametric"`` : Assumes returns are normally distributed.
        - ``"historical"`` : Uses the empirical quantile of *returns*.
        - ``"monte_carlo"`` : Draws from a fitted normal distribution.

        Default is ``"parametric"``.
    n_sims : int, optional
        Number of Monte Carlo draws (only used when ``method="monte_carlo"``).
        Default is ``10000``.
    random_state : int or None, optional
        Random seed for Monte Carlo reproducibility.  Default is ``None``.

    Returns
    -------
    float
        VaR expressed as a negative number (loss).  For example, ``-0.02``
        means the portfolio is expected to lose no more than 2 % with
        probability *confidence*.

    Notes
    -----
    **Parametric (Gaussian) VaR**

    Assumes returns :math:`R \sim \mathcal{N}(\mu, \sigma^2)`:

    .. math::

        \text{VaR}_c = \mu - z_c \, \sigma

    where :math:`z_c = \Phi^{-1}(c)` with :math:`c` the confidence level,
    and :math:`\mu`, :math:`\sigma` are the sample mean and standard
    deviation of *returns*.  For :math:`c = 0.95`,
    :math:`z_c \approx 1.645`, giving
    :math:`\text{VaR} \approx \mu - 1.645\,\sigma`.

    Equivalently, setting :math:`\alpha = 1 - c`:

    .. math::

        \text{VaR}_c = \mu + z_\alpha \, \sigma

    where :math:`z_\alpha = \Phi^{-1}(\alpha)` is negative.

    **Historical VaR**

    .. math::

        \text{VaR}_c = Q_\alpha(R)

    where :math:`Q_\alpha` is the empirical :math:`\alpha`-quantile.

    **Monte Carlo VaR**

    Simulate :math:`N` returns from :math:`\mathcal{N}(\hat\mu,
    \hat\sigma^2)` and take the :math:`\alpha`-quantile.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, 500), index=idx)
    >>> value_at_risk(ret, confidence=0.95, method="parametric")  # doctest: +SKIP
    -0.016...
    """
    if not isinstance(returns, _pd.Series):
        raise TypeError("returns must be a pandas Series")

    _valid = {"parametric", "historical", "monte_carlo"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    r = returns.dropna().values.astype(float)
    alpha = 1.0 - confidence

    if len(r) == 0:
        return _np.nan

    if method == "parametric":
        mu = _np.mean(r)
        sigma = _np.std(r, ddof=1)
        return float(mu + _norm.ppf(alpha) * sigma)

    elif method == "historical":
        return float(_np.percentile(r, alpha * 100))

    else:  # monte_carlo
        rng = _np.random.default_rng(random_state)
        mu = _np.mean(r)
        sigma = _np.std(r, ddof=1)
        sims = rng.normal(mu, sigma, n_sims)
        return float(_np.percentile(sims, alpha * 100))


def cvar(
    returns: _pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    r"""
    Compute Conditional Value at Risk (CVaR), also known as Expected
    Shortfall.

    CVaR is the expected loss given that the loss exceeds the VaR threshold.
    It is a coherent risk measure and captures tail risk more completely
    than VaR alone.

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic returns.
    confidence : float, optional
        Confidence level.  Default is ``0.95``.
    method : str, optional
        Estimation method, one of ``"parametric"`` or ``"historical"``.
        Default is ``"historical"``.

    Returns
    -------
    float
        CVaR expressed as a negative number (expected loss in the tail).

    Notes
    -----
    **Parametric (Gaussian) CVaR**

    Under the normality assumption:

    .. math::

        \text{ES}_c = \mu
            - \sigma \, \frac{\varphi(z_\alpha)}{1 - c}

    where :math:`\varphi(\cdot)` is the standard normal PDF,
    :math:`z_\alpha = \Phi^{-1}(1-c)`, and :math:`\mu`, :math:`\sigma`
    are the sample mean and standard deviation.  The quantity
    :math:`-\mu + \sigma\,\varphi(z_\alpha)/(1-c)` is the loss-convention
    Expected Shortfall; negating it yields the P&L-convention value
    returned by this function.

    For :math:`c = 0.95`: :math:`z_\alpha \approx -1.645`,
    :math:`\varphi(-1.645) \approx 0.1031`, so

    .. math::

        \text{ES}_{0.95} \approx \mu - 2.063\,\sigma

    **Historical CVaR**

    .. math::

        \text{ES}_c = \mathbb{E}[R \mid R \le \text{VaR}_c]

    i.e. the mean of all returns at or below the VaR threshold.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, 500), index=idx)
    >>> cvar(ret, confidence=0.95, method="historical")  # doctest: +SKIP
    -0.022...
    """
    if not isinstance(returns, _pd.Series):
        raise TypeError("returns must be a pandas Series")

    _valid = {"parametric", "historical"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    r = returns.dropna().values.astype(float)
    alpha = 1.0 - confidence

    if len(r) == 0:
        return _np.nan

    if method == "parametric":
        mu = _np.mean(r)
        sigma = _np.std(r, ddof=1)
        z_alpha = _norm.ppf(alpha)
        return float(mu - sigma * _norm.pdf(z_alpha) / alpha)

    else:  # historical
        var_threshold = _np.percentile(r, alpha * 100)
        tail = r[r <= var_threshold]
        if len(tail) == 0:
            return float(var_threshold)
        return float(_np.mean(tail))


def cfar(
    cash_flows: Union[_pd.DataFrame, _pd.Series],
    confidence: float = 0.95,
    horizon: Optional[int] = None,
    method: str = "historical",
) -> dict:
    r"""
    Cash Flow at Risk (CFaR).

    CFaR extends Value at Risk to projected cash-flow streams over longer
    horizons.  Instead of measuring the worst single-period return, CFaR
    measures the worst-case aggregate cash flow over a multi-period horizon.

    Parameters
    ----------
    cash_flows : pd.DataFrame or pd.Series
        - **DataFrame** : each column represents one simulated cash-flow
          scenario (e.g. from Monte Carlo simulation).  Rows are time
          periods.  This is the typical output of a stochastic cash-flow
          model.
        - **Series** : a historical time series of observed cash flows.  The
          function computes rolling sums of length *horizon* and applies VaR
          to those rolling totals.
    confidence : float, optional
        Confidence level.  Default is ``0.95``.
    horizon : int or None, optional
        Number of periods to aggregate.  If ``None``, all periods are summed
        (i.e. the full projection horizon is used).
    method : str, optional
        VaR estimation method for the aggregated cash flows, either
        ``"historical"`` (empirical quantile) or ``"parametric"`` (Gaussian).
        Default is ``"historical"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"cfar"`` : float -- the Cash Flow at Risk value.
        - ``"expected_cf"`` : float -- mean total cash flow.
        - ``"worst_case_cf"`` : float -- minimum total cash flow observed.
        - ``"n_scenarios"`` : int -- number of scenarios used.

    Notes
    -----
    Cash Flow at Risk is conceptually identical to VaR, but applied to
    aggregated cash flows rather than single-period returns:

    .. math::

        \text{CFaR}_\alpha = Q_\alpha\!\left(\sum_{t=1}^{H} CF_t\right)

    where :math:`H` is the *horizon* and :math:`Q_\alpha` is the
    :math:`\alpha`-quantile of the cash-flow distribution with
    :math:`\alpha = 1 - c`.

    For **simulated scenarios** (DataFrame input), each column is summed
    over the first *horizon* rows, producing a distribution of total cash
    flows.  CFaR is the :math:`\alpha`-quantile of that distribution.

    For **historical data** (Series input), a rolling sum of window size
    *horizon* is computed, and CFaR is the :math:`\alpha`-quantile of those
    rolling totals.

    References
    ----------
    Stein, J.C., Usher, S.E., LaGattuta, D. and Youngen, J. (2001).
    "A Comparables Approach to Measuring Cashflow-at-Risk for Non-Financial
    Firms." *Journal of Applied Corporate Finance*, 13(4), 100-109.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # 100 simulated 12-month cash flow paths
    >>> cf = pd.DataFrame(rng.normal(10, 3, (12, 100)))
    >>> result = cfar(cf, confidence=0.95, horizon=12)
    >>> result["cfar"] < result["expected_cf"]
    True
    """
    if not isinstance(cash_flows, (_pd.DataFrame, _pd.Series)):
        raise TypeError(
            "cash_flows must be a pd.DataFrame (simulated paths) or "
            f"pd.Series (historical), got {type(cash_flows).__name__}"
        )
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    _valid = {"parametric", "historical"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")

    alpha = 1.0 - confidence

    if isinstance(cash_flows, _pd.DataFrame):
        # Simulated scenarios: each column is one path
        if horizon is not None:
            cf_slice = cash_flows.iloc[:horizon]
        else:
            cf_slice = cash_flows
        # Sum each column to get total cash flow per scenario
        totals = cf_slice.sum(axis=0).values.astype(float)

    else:  # pd.Series
        # Historical data: rolling sum
        if horizon is None:
            # Single total -- can't compute distribution, treat as one scenario
            totals = _np.array([float(cash_flows.sum())])
        else:
            rolling_totals = cash_flows.rolling(
                window=horizon, min_periods=horizon
            ).sum()
            totals = rolling_totals.dropna().values.astype(float)

    if len(totals) == 0:
        raise ValueError("Not enough data to compute CFaR")

    if method == "parametric":
        mu = _np.mean(totals)
        sigma = _np.std(totals, ddof=1) if len(totals) > 1 else 0.0
        cfar_value = float(mu + _norm.ppf(alpha) * sigma)
    else:  # historical
        cfar_value = float(_np.percentile(totals, alpha * 100))

    return {
        "cfar": cfar_value,
        "expected_cf": float(_np.mean(totals)),
        "worst_case_cf": float(_np.min(totals)),
        "n_scenarios": len(totals),
    }


def npv_at_risk(
    cash_flows: Union[_pd.DataFrame, _pd.Series],
    discount_rate: Union[float, _pd.Series, _pd.DataFrame],
    confidence: float = 0.95,
    method: str = "historical",
) -> dict:
    r"""
    NPV at Risk -- Value at Risk applied to a distribution of Net Present
    Values.

    NPVaR extends VaR to multi-period discounted cash-flow analysis.
    Given simulated or stochastic cash-flow paths and discount rates, it
    computes the NPV of each scenario and then reports VaR and CVaR on the
    resulting NPV distribution.

    Parameters
    ----------
    cash_flows : pd.DataFrame or pd.Series
        - **DataFrame** : each column is one simulated cash-flow path, rows
          are time periods (:math:`t = 0, 1, \ldots, T-1`).  Row 0 is
          typically the initial investment (negative).  Each column is an
          independent scenario.
        - **Series** : a single deterministic cash-flow stream.  Combine
          with a stochastic *discount_rate* (supplied as a DataFrame) to
          obtain an NPV distribution.
    discount_rate : float, pd.Series, or pd.DataFrame
        - **float** : constant discount rate applied to all periods and all
          scenarios.
        - **pd.Series** : period-specific discount rates (same length as
          number of rows in *cash_flows*), applied to all scenarios.
        - **pd.DataFrame** : scenario-specific discount rates (same shape as
          *cash_flows*), allowing both period- and scenario-variation.
    confidence : float, optional
        Confidence level for VaR / CVaR calculations.  Default is ``0.95``.
    method : str, optional
        Estimation method for the risk metrics computed on the NPV
        distribution, one of ``"historical"`` or ``"parametric"``.
        Default is ``"historical"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"npv_at_risk"`` : float -- VaR of the NPV distribution.
        - ``"npv_cvar"`` : float -- CVaR (Expected Shortfall) of the NPV
          distribution.
        - ``"expected_npv"`` : float -- mean NPV across scenarios.
        - ``"std_npv"`` : float -- standard deviation of NPV.
        - ``"npv_distribution"`` : pd.Series -- individual NPV values (one
          per simulation).
        - ``"percentiles"`` : dict -- percentile map with keys ``"5%"``,
          ``"25%"``, ``"50%"``, ``"75%"``, ``"95%"``.

    Notes
    -----
    For each scenario (column) :math:`j`, the net present value is:

    .. math::

        \text{NPV}^{(j)} = \sum_{t=0}^{T-1}
            \frac{CF_{t}^{(j)}}{(1 + r_{t}^{(j)})^{t}}

    where :math:`r_{t}^{(j)}` is the discount rate for period :math:`t` in
    scenario :math:`j` (a constant when *discount_rate* is a scalar).  Note
    that at :math:`t = 0` the discount factor is 1 (no discounting).

    Then:

    .. math::

        \text{NPVaR}_\alpha = Q_\alpha(\text{NPV}_1, \ldots, \text{NPV}_M)

    .. math::

        \text{NPV-CVaR}_\alpha
            = \mathbb{E}[\text{NPV} \mid \text{NPV} \le \text{NPVaR}_\alpha]

    References
    ----------
    Ye, S. and Tiong, R.L.K. (2000).  "NPV-at-Risk Method in
    Infrastructure Project Investment Evaluation."  *Journal of
    Construction Engineering and Management*, 126(3), 227-233.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # Initial investment of -100, then 12 months of stochastic CFs
    >>> cf_data = np.vstack([
    ...     np.full(1000, -100.0),
    ...     rng.normal(12, 3, (12, 1000)),
    ... ])
    >>> cf = pd.DataFrame(cf_data)
    >>> result = npv_at_risk(cf, discount_rate=0.10 / 12, confidence=0.95)
    >>> result["expected_npv"]  # doctest: +SKIP
    35.5...
    """
    if not isinstance(cash_flows, (_pd.DataFrame, _pd.Series)):
        raise TypeError(
            f"cash_flows must be pd.DataFrame or pd.Series, "
            f"got {type(cash_flows).__name__}"
        )
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    _valid = {"parametric", "historical"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")

    # Convert cash_flows to 2D array: (n_periods, n_scenarios)
    if isinstance(cash_flows, _pd.Series):
        cf_arr = cash_flows.values.astype(float).reshape(-1, 1)
    else:
        cf_arr = cash_flows.values.astype(float)

    n_periods, n_scenarios = cf_arr.shape
    periods = _np.arange(n_periods, dtype=float)

    # Build discount factor array: (n_periods, n_scenarios)
    if isinstance(discount_rate, (int, float)):
        discount_factors = 1.0 / (1.0 + float(discount_rate)) ** periods
        discount_factors = discount_factors.reshape(-1, 1)  # broadcast
    elif isinstance(discount_rate, _pd.Series):
        r_arr = discount_rate.values[:n_periods].astype(float)
        discount_factors = 1.0 / (1.0 + r_arr) ** periods
        discount_factors = discount_factors.reshape(-1, 1)
    elif isinstance(discount_rate, _pd.DataFrame):
        r_arr = discount_rate.values[:n_periods, :n_scenarios].astype(float)
        discount_factors = 1.0 / (1.0 + r_arr) ** periods.reshape(-1, 1)
    else:
        raise TypeError(
            f"discount_rate must be float, pd.Series, or pd.DataFrame, "
            f"got {type(discount_rate).__name__}"
        )

    # Compute NPV for each scenario
    npvs = (cf_arr * discount_factors).sum(axis=0)

    # Wrap in a Series for the risk metric functions
    npv_series = _pd.Series(npvs, dtype=float)

    # VaR and CVaR on the NPV distribution
    npv_var = value_at_risk(npv_series, confidence=confidence, method=method)
    npv_cvar_val = cvar(
        npv_series, confidence=confidence,
        method=method if method in {"parametric", "historical"} else "historical",
    )

    return {
        "npv_at_risk": float(npv_var),
        "npv_cvar": float(npv_cvar_val),
        "expected_npv": float(_np.mean(npvs)),
        "std_npv": float(_np.std(npvs, ddof=1)) if n_scenarios > 1 else 0.0,
        "npv_distribution": npv_series,
        "percentiles": {
            "5%": float(_np.percentile(npvs, 5)),
            "25%": float(_np.percentile(npvs, 25)),
            "50%": float(_np.percentile(npvs, 50)),
            "75%": float(_np.percentile(npvs, 75)),
            "95%": float(_np.percentile(npvs, 95)),
        },
    }


def rolling_var(
    returns: _pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> _pd.Series:
    r"""
    Rolling Value at Risk over a sliding window.

    For each position in the series, compute VaR using the preceding
    *window* observations.

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic returns with a ``DatetimeIndex``.
    window : int, optional
        Rolling window size in periods.  Default is ``252`` (~1 trading
        year).
    confidence : float, optional
        Confidence level.  Default is ``0.95``.
    method : str, optional
        ``"parametric"`` or ``"historical"``.  Default is ``"historical"``.

    Returns
    -------
    pd.Series
        Rolling VaR values (negative numbers), indexed identically to
        *returns*.  The first ``window - 1`` entries are ``NaN``.

    Notes
    -----
    For each window ending at time :math:`t`:

    - **Historical** :
      :math:`\text{VaR}_t = Q_\alpha(R_{t-w+1}, \ldots, R_t)`
    - **Parametric** :
      :math:`\text{VaR}_t = \hat\mu_t + z_\alpha \, \hat\sigma_t`

    where :math:`\alpha = 1 - c`, :math:`z_\alpha = \Phi^{-1}(\alpha)`,
    and :math:`\hat\mu_t`, :math:`\hat\sigma_t` are the rolling sample
    mean and standard deviation.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0, 0.01, 500), index=idx)
    >>> rv = rolling_var(ret, window=63, confidence=0.95)
    >>> rv.dropna().shape[0]
    438
    """
    if not isinstance(returns, _pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, _pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    _valid = {"parametric", "historical"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    alpha = 1.0 - confidence
    n = len(returns)
    values = returns.values

    if method == "historical":
        result = _np.full(n, _np.nan)
        for t in range(window - 1, n):
            win = values[t - window + 1 : t + 1]
            result[t] = _np.percentile(win, alpha * 100)

    else:  # parametric
        z = _norm.ppf(alpha)
        roll_mean = returns.rolling(
            window=window, min_periods=window
        ).mean()
        roll_std = returns.rolling(
            window=window, min_periods=window
        ).std(ddof=1)
        result = (roll_mean + z * roll_std).values

    return _pd.Series(result, index=returns.index, name="rolling_var")


def rolling_cvar(
    returns: _pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> _pd.Series:
    r"""
    Rolling Conditional Value at Risk (Expected Shortfall) over a sliding
    window.

    For each position in the series, compute CVaR using the preceding
    *window* observations.

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic returns with a ``DatetimeIndex``.
    window : int, optional
        Rolling window size in periods.  Default is ``252``.
    confidence : float, optional
        Confidence level.  Default is ``0.95``.
    method : str, optional
        ``"parametric"`` or ``"historical"``.  Default is ``"historical"``.

    Returns
    -------
    pd.Series
        Rolling CVaR values (negative numbers), indexed identically to
        *returns*.  The first ``window - 1`` entries are ``NaN``.

    Notes
    -----
    For each window ending at time :math:`t`:

    - **Historical** :
      :math:`\text{CVaR}_t = \mathbb{E}[R_i \mid R_i \le \text{VaR}_t]`,
      i.e. the mean of returns in the window that fall at or below the VaR
      threshold.
    - **Parametric** :

      .. math::

          \text{CVaR}_t = \hat\mu_t
              - \hat\sigma_t \,
              \frac{\varphi(z_\alpha)}{\alpha}

      where :math:`\alpha = 1 - c`, :math:`z_\alpha = \Phi^{-1}(\alpha)`,
      and :math:`\varphi` is the standard normal PDF.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0, 0.01, 500), index=idx)
    >>> rc = rolling_cvar(ret, window=63, confidence=0.95)
    >>> rc.dropna().shape[0]
    438
    """
    if not isinstance(returns, _pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, _pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    _valid = {"parametric", "historical"}
    if method not in _valid:
        raise ValueError(f"method must be one of {_valid!r}, got {method!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    alpha = 1.0 - confidence
    n = len(returns)
    values = returns.values
    result = _np.full(n, _np.nan)

    if method == "historical":
        for t in range(window - 1, n):
            win = values[t - window + 1 : t + 1]
            threshold = _np.percentile(win, alpha * 100)
            tail = win[win <= threshold]
            result[t] = _np.mean(tail) if len(tail) > 0 else threshold

    else:  # parametric
        z_alpha = _norm.ppf(alpha)
        phi_z = _norm.pdf(z_alpha)
        roll_mean = returns.rolling(
            window=window, min_periods=window
        ).mean()
        roll_std = returns.rolling(
            window=window, min_periods=window
        ).std(ddof=1)
        result = (roll_mean - roll_std * phi_z / alpha).values

    return _pd.Series(result, index=returns.index, name="rolling_cvar")


def parametric_var_portfolio(
    weights: Union[_np.ndarray, _pd.Series],
    cov_matrix: Union[_np.ndarray, _pd.DataFrame],
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    r"""
    Portfolio VaR using the parametric (variance-covariance) method.

    Computes the portfolio VaR under the assumption that asset returns are
    jointly normally distributed.  This method uses only the covariance
    structure (zero expected return assumption), which is standard for
    short-horizon portfolio risk measurement.

    Parameters
    ----------
    weights : np.ndarray or pd.Series
        Portfolio weights.  Must have the same length as the number of
        assets (rows/columns of *cov_matrix*).
    cov_matrix : np.ndarray or pd.DataFrame
        Covariance matrix of asset returns, shape ``(n, n)``.
    confidence : float, optional
        Confidence level.  Default is ``0.95``.
    portfolio_value : float, optional
        Total portfolio notional value.  Default is ``1.0``
        (return-based VaR).

    Returns
    -------
    float
        Portfolio VaR as a **negative** number (profit/loss convention).
        Multiply by ``-1`` to obtain the loss magnitude.

    Notes
    -----
    Under the Gaussian assumption the portfolio standard deviation is:

    .. math::

        \sigma_p = \sqrt{\mathbf{w}^\top \, \Sigma \, \mathbf{w}}

    and VaR at confidence level :math:`c` is:

    .. math::

        \text{VaR}_c = z_\alpha \; \sigma_p \; V

    where :math:`z_\alpha = \Phi^{-1}(1-c)` (a negative number for
    :math:`c > 0.5`) and :math:`V` is the portfolio value.  This assumes
    zero expected return (:math:`\mu_p = 0`).

    For :math:`c = 0.95`:

    .. math::

        \text{VaR}_{0.95} = \Phi^{-1}(0.05) \; \sigma_p \; V
            \approx -1.645 \, \sigma_p \, V

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.6, 0.4])
    >>> cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    >>> parametric_var_portfolio(w, cov, confidence=0.95, portfolio_value=1e6)
    ... # doctest: +SKIP
    -264038.7...
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    w = _np.asarray(weights, dtype=float).ravel()
    cov = _np.asarray(cov_matrix, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix")
    if w.shape[0] != cov.shape[0]:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match cov_matrix dimension "
            f"({cov.shape[0]})"
        )

    port_var = float(w @ cov @ w)
    port_sigma = _np.sqrt(port_var)

    alpha = 1.0 - confidence
    z_alpha = _norm.ppf(alpha)

    return float(z_alpha * port_sigma * portfolio_value)
