# Rolling Analytics Functions

from typing import Union

import pandas as _pd
import numpy as _np

__all__ = [
    "rolling_sharpe",
    "rolling_beta",
    "rolling_correlation",
    "rolling_volatility",
    "rolling_skewness",
    "rolling_kurtosis",
    "rolling_drawdown",
    "rolling_sortino",
]

# Mapping from pandas frequency strings to annualization factors
_FREQ_SCALE = {
    "D": 252, "B": 252,
    "W": 52,
    "M": 12, "ME": 12, "MS": 12,
    "Q": 4, "QE": 4, "QS": 4,
    "Y": 1, "YE": 1, "YS": 1, "A": 1, "AS": 1,
}


def _infer_scale(returns: _pd.Series) -> int:
    r"""
    Infer annualization factor from a return series' DatetimeIndex frequency.

    Inspects the ``.freq`` attribute of the series index and maps it to the
    corresponding number of periods per year using ``_FREQ_SCALE``.  Falls back
    to 252 (daily) when the frequency cannot be determined.

    Parameters
    ----------
    returns : pd.Series
        A return series whose index is ideally a ``DatetimeIndex`` with a
        ``freq`` attribute set.

    Returns
    -------
    int
        Annualization factor (e.g. 252 for daily, 12 for monthly).
    """
    if not isinstance(returns.index, _pd.DatetimeIndex):
        return 252

    freq = returns.index.freq
    if freq is None:
        return 252

    freq_str = getattr(freq, "freqstr", None) or getattr(freq, "name", None) or str(freq)

    if freq_str in _FREQ_SCALE:
        return _FREQ_SCALE[freq_str]

    # Try prefix match for frequencies like 'W-FRI'
    if freq_str and freq_str[0] in _FREQ_SCALE:
        return _FREQ_SCALE[freq_str[0]]

    return 252


def _validate_series(x, name: str = "returns") -> None:
    r"""
    Validate that *x* is a ``pd.Series``.

    Parameters
    ----------
    x : any
        Value to validate.
    name : str
        Parameter name used in the error message.

    Raises
    ------
    TypeError
        If *x* is not a ``pd.Series``.
    """
    if not isinstance(x, _pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(x).__name__}")


def rolling_sharpe(
    returns: _pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Compute a rolling Sharpe ratio over a fixed window.

    For each rolling window of length *window* the Sharpe ratio is computed as

    .. math::

        \text{Sharpe}_t = \frac{\bar{e}_t}{\sigma_{e,t}}

    where :math:`e_i = R_i - R_f` are the excess returns inside the window,
    :math:`\bar{e}_t` is their mean, and :math:`\sigma_{e,t}` their sample
    standard deviation.

    When *annualize* is ``True`` the ratio is scaled by :math:`\sqrt{s}` where
    *s* is the annualization factor inferred from the index frequency (or 252
    by default):

    .. math::

        \text{Sharpe}^{\text{ann}}_t = \text{Sharpe}_t \times \sqrt{s}

    Parameters
    ----------
    returns : pd.Series
        Periodic (e.g. daily) simple returns with a ``DatetimeIndex``.
    window : int, default 252
        Rolling window size in periods (~1 trading year for daily data).
    risk_free_rate : float, default 0.0
        Periodic risk-free rate expressed in the same frequency as *returns*.
    annualize : bool, default True
        Whether to annualize the ratio.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio with ``NaN`` for the initial *window* - 1
        periods.

    Notes
    -----
    A rolling window with constant excess returns produces a standard
    deviation of zero; the Sharpe ratio for that window is ``NaN``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, len(idx)), index=idx)
    >>> sr = rolling_sharpe(ret, window=63)
    """
    _validate_series(returns, "returns")
    excess = returns - risk_free_rate

    roll_mean = excess.rolling(window=window, min_periods=window).mean()
    roll_std = excess.rolling(window=window, min_periods=window).std(ddof=1)

    sharpe = roll_mean / roll_std

    if annualize:
        scale = _infer_scale(returns)
        sharpe = sharpe * _np.sqrt(scale)

    return sharpe.rename("rolling_sharpe")


def rolling_beta(
    asset_returns: _pd.Series,
    benchmark_returns: _pd.Series,
    window: int = 252,
) -> _pd.Series:
    r"""
    Compute a rolling CAPM beta of an asset relative to a benchmark.

    For each rolling window of length *window*:

    .. math::

        \beta_t = \frac{\operatorname{Cov}(R_a, R_b)}{\operatorname{Var}(R_b)}

    where :math:`R_a` and :math:`R_b` are asset and benchmark returns inside
    the window, respectively.

    Parameters
    ----------
    asset_returns : pd.Series
        Periodic returns of the asset.
    benchmark_returns : pd.Series
        Periodic returns of the benchmark (same frequency / alignment as
        *asset_returns*).
    window : int, default 252
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling beta with ``NaN`` for initial *window* - 1 periods.

    Notes
    -----
    Both series are first aligned on their shared index and any rows with
    ``NaN`` in either series are dropped before rolling.

    When the benchmark variance in a window is zero the beta is ``NaN``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> bench = pd.Series(np.random.normal(0.0003, 0.01, len(idx)), index=idx)
    >>> asset = 1.2 * bench + np.random.normal(0, 0.005, len(idx))
    >>> asset = pd.Series(asset, index=idx)
    >>> b = rolling_beta(asset, bench, window=63)
    """
    _validate_series(asset_returns, "asset_returns")
    _validate_series(benchmark_returns, "benchmark_returns")

    # Align on shared index and drop NaN pairs
    combined = _pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("bench")],
        axis=1,
    ).dropna()

    asset_al = combined["asset"]
    bench_al = combined["bench"]

    # Rolling covariance and variance
    roll_cov = asset_al.rolling(window=window, min_periods=window).cov(bench_al)
    roll_var = bench_al.rolling(window=window, min_periods=window).var(ddof=1)

    beta = roll_cov / roll_var

    return beta.rename("rolling_beta")


def rolling_correlation(
    x: _pd.Series,
    y: _pd.Series,
    window: int = 252,
) -> _pd.Series:
    r"""
    Compute a rolling Pearson correlation between two series.

    For each rolling window of length *window*:

    .. math::

        \rho_t = \frac{\operatorname{Cov}(X, Y)}
                      {\sigma_X \, \sigma_Y}

    Parameters
    ----------
    x : pd.Series
        First series.
    y : pd.Series
        Second series (same frequency / alignment as *x*).
    window : int, default 252
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling Pearson correlation with ``NaN`` for initial *window* - 1
        periods.

    Notes
    -----
    Both series are aligned on their shared index and rows with ``NaN`` in
    either series are dropped before computing the rolling statistic.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> x = pd.Series(np.random.normal(0, 1, len(idx)), index=idx)
    >>> y = 0.5 * x + pd.Series(np.random.normal(0, 1, len(idx)), index=idx)
    >>> corr = rolling_correlation(x, y, window=63)
    """
    _validate_series(x, "x")
    _validate_series(y, "y")

    combined = _pd.concat(
        [x.rename("x"), y.rename("y")],
        axis=1,
    ).dropna()

    corr = combined["x"].rolling(window=window, min_periods=window).corr(combined["y"])

    return corr.rename("rolling_correlation")


def rolling_volatility(
    returns: _pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Compute rolling (optionally annualized) volatility.

    For each rolling window of length *window*:

    .. math::

        \sigma_t = \sqrt{\frac{1}{n-1}\sum_{i \in W_t}(R_i - \bar{R})^2}

    When *annualize* is ``True``:

    .. math::

        \sigma^{\text{ann}}_t = \sigma_t \times \sqrt{s}

    where *s* is the annualization factor inferred from the index frequency (or
    252 by default).

    Parameters
    ----------
    returns : pd.Series
        Periodic (e.g. daily) simple returns with a ``DatetimeIndex``.
    window : int, default 21
        Rolling window size in periods (~1 trading month for daily data).
    annualize : bool, default True
        Whether to annualize the volatility.

    Returns
    -------
    pd.Series
        Rolling volatility with ``NaN`` for initial *window* - 1 periods.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, len(idx)), index=idx)
    >>> vol = rolling_volatility(ret, window=21)
    """
    _validate_series(returns, "returns")

    vol = returns.rolling(window=window, min_periods=window).std(ddof=1)

    if annualize:
        scale = _infer_scale(returns)
        vol = vol * _np.sqrt(scale)

    return vol.rename("rolling_volatility")


def rolling_skewness(
    returns: _pd.Series,
    window: int = 252,
) -> _pd.Series:
    r"""
    Compute rolling skewness of returns.

    For each rolling window the sample skewness (Fisher's definition) is:

    .. math::

        g_1 = \frac{m_3}{m_2^{3/2}}
            = \frac{\frac{1}{n}\sum_{i}(R_i - \bar{R})^3}
                   {\left[\frac{1}{n}\sum_{i}(R_i - \bar{R})^2\right]^{3/2}}

    Pandas applies the bias-corrected adjustment by default.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns.
    window : int, default 252
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling skewness with ``NaN`` for initial *window* - 1 periods.

    Notes
    -----
    A perfectly symmetric distribution has skewness of 0. Negative skewness
    indicates a longer / fatter left tail, which is typical of equity returns.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    >>> sk = rolling_skewness(ret, window=126)
    """
    _validate_series(returns, "returns")

    skew = returns.rolling(window=window, min_periods=window).skew()

    return skew.rename("rolling_skewness")


def rolling_kurtosis(
    returns: _pd.Series,
    window: int = 252,
) -> _pd.Series:
    r"""
    Compute rolling excess kurtosis of returns.

    For each rolling window the excess kurtosis is:

    .. math::

        \kappa_{\text{excess}} = \frac{m_4}{m_2^{2}} - 3

    where :math:`m_k` is the *k*-th central moment of the sample.  Pandas'
    ``rolling().kurt()`` already returns excess kurtosis (Fisher's definition),
    so no additional subtraction is needed.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns.
    window : int, default 252
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling excess kurtosis with ``NaN`` for initial *window* - 1
        periods.

    Notes
    -----
    Excess kurtosis of zero corresponds to a normal distribution (mesokurtic).
    Positive values indicate heavier tails (leptokurtic), negative values
    indicate lighter tails (platykurtic).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    >>> ku = rolling_kurtosis(ret, window=126)
    """
    _validate_series(returns, "returns")

    kurt = returns.rolling(window=window, min_periods=window).kurt()

    return kurt.rename("rolling_kurtosis")


def rolling_drawdown(
    returns: _pd.Series,
    window: int = 252,
) -> _pd.Series:
    r"""
    Compute rolling maximum drawdown over a fixed window.

    For each window :math:`[t - w + 1, \; t]` the cumulative wealth curve is
    constructed from the periodic returns:

    .. math::

        W_i = \prod_{j=t-w+1}^{i} (1 + R_j), \quad i \in [t-w+1, \; t]

    The maximum drawdown within that window is:

    .. math::

        \text{MDD}_t = \min_{i \in W_t}
            \left(\frac{W_i}{\max_{j \le i} W_j} - 1\right)

    Parameters
    ----------
    returns : pd.Series
        Periodic (e.g. daily) simple returns with a ``DatetimeIndex``.
    window : int, default 252
        Rolling window size in periods.

    Returns
    -------
    pd.Series
        Rolling maximum drawdown expressed as negative values (e.g. -0.15
        means a 15 % peak-to-trough decline).  ``NaN`` for initial
        *window* - 1 periods.

    Notes
    -----
    The computation iterates over each window position because there is no
    vectorised built-in for peak-to-trough drawdown.  For very long series or
    very large windows this may be slow.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, len(idx)), index=idx)
    >>> dd = rolling_drawdown(ret, window=63)
    """
    _validate_series(returns, "returns")

    n = len(returns)
    values = returns.values
    result = _np.full(n, _np.nan)

    for t in range(window - 1, n):
        # Slice the window of returns
        win_ret = values[t - window + 1 : t + 1]

        # Build cumulative wealth inside the window (starting at 1.0)
        cum_wealth = _np.cumprod(1.0 + win_ret)

        # Running maximum of cumulative wealth
        running_max = _np.maximum.accumulate(cum_wealth)

        # Drawdown at each point in the window
        drawdowns = cum_wealth / running_max - 1.0

        # Maximum drawdown (most negative value)
        result[t] = _np.min(drawdowns)

    return _pd.Series(result, index=returns.index, name="rolling_drawdown")


def rolling_sortino(
    returns: _pd.Series,
    window: int = 252,
    target_return: float = 0.0,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Compute a rolling Sortino ratio over a fixed window.

    For each rolling window the Sortino ratio is:

    .. math::

        \text{Sortino}_t =
            \frac{\bar{e}_t}{\sigma_{\text{down},t}}

    where :math:`e_i = R_i - R_{\text{target}}` are excess returns and the
    downside deviation is:

    .. math::

        \sigma_{\text{down},t}
            = \sqrt{\frac{1}{n}\sum_{i \in W_t}
              \left[\min(e_i, 0)\right]^2}

    When *annualize* is ``True``:

    .. math::

        \text{Sortino}^{\text{ann}}_t
            = \text{Sortino}_t \times \sqrt{s}

    where *s* is the annualization factor.

    Parameters
    ----------
    returns : pd.Series
        Periodic (e.g. daily) simple returns with a ``DatetimeIndex``.
    window : int, default 252
        Rolling window size in periods.
    target_return : float, default 0.0
        Minimum acceptable return expressed in the same frequency as
        *returns*.
    annualize : bool, default True
        Whether to annualize the ratio.

    Returns
    -------
    pd.Series
        Rolling Sortino ratio with ``NaN`` for initial *window* - 1
        periods.

    Notes
    -----
    When no returns in the window fall below the target the downside
    deviation is zero, producing ``inf`` (or ``NaN`` if the mean excess
    return is also zero).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    >>> ret = pd.Series(np.random.normal(0.0004, 0.01, len(idx)), index=idx)
    >>> so = rolling_sortino(ret, window=63)
    """
    _validate_series(returns, "returns")

    excess = returns - target_return

    roll_mean = excess.rolling(window=window, min_periods=window).mean()

    # Downside: clip excess returns at zero, square, take rolling mean, sqrt
    downside_sq = excess.clip(upper=0.0) ** 2
    downside_dev = downside_sq.rolling(window=window, min_periods=window).mean().pow(0.5)

    sortino = roll_mean / downside_dev

    if annualize:
        scale = _infer_scale(returns)
        sortino = sortino * _np.sqrt(scale)

    return sortino.rename("rolling_sortino")
