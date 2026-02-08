# Volatility Analytics Functions

from typing import Union, List, Optional

import pandas as _pd
import numpy as _np

__all__ = [
    "vol_cone",
    "vol_term_structure",
    "realized_vol",
    "parkinson_vol",
    "garman_klass_vol",
    "yang_zhang_vol",
]

# Mapping from pandas frequency strings to trading periods per year
_FREQ_SCALE = {
    "D": 252, "B": 252,
    "W": 52,
    "M": 12, "ME": 12, "MS": 12,
    "Q": 4, "QE": 4, "QS": 4,
    "Y": 1, "YE": 1, "YS": 1, "A": 1, "AS": 1,
}


def _annualization_factor(returns: _pd.Series) -> int:
    r"""
    Infer the annualization factor from the frequency of a returns series.

    Inspects the ``DatetimeIndex.freq`` attribute and maps it to a standard
    number of trading periods per year.  Falls back to **252** (daily) when
    the frequency cannot be determined.

    Parameters
    ----------
    returns : pd.Series
        A time series of returns whose index is a ``DatetimeIndex``.

    Returns
    -------
    int
        The number of periods per year (e.g. 252 for daily, 52 for weekly).
    """
    if not isinstance(returns.index, _pd.DatetimeIndex):
        return 252

    freq = returns.index.freq
    if freq is None:
        return 252

    freq_str = (
        getattr(freq, "freqstr", None)
        or getattr(freq, "name", None)
        or str(freq)
    )

    if freq_str in _FREQ_SCALE:
        return _FREQ_SCALE[freq_str]

    # Prefix match for frequencies like 'W-FRI'
    if freq_str and freq_str[0] in _FREQ_SCALE:
        return _FREQ_SCALE[freq_str[0]]

    return 252


def vol_cone(
    returns: _pd.Series,
    windows: Optional[List[int]] = None,
    quantiles: Optional[List[float]] = None,
) -> _pd.DataFrame:
    r"""
    Volatility cone showing the distribution of realized volatility across
    different lookback windows.

    A volatility cone provides context for whether current realized volatility
    is high or low relative to its historical distribution at each horizon.
    For each rolling window size *w*, the function computes the full history of
    rolling realized volatilities and then reports selected quantiles together
    with the most recent (current) value.

    The rolling realized volatility at time *t* for window *w* is:

    .. math::

        \sigma_{t}^{(w)} = \operatorname{std}(R_{t-w+1}, \ldots, R_{t})
            \cdot \sqrt{N}

    where :math:`N` is the annualization factor (252 for daily data).

    Parameters
    ----------
    returns : pd.Series
        A time series of asset returns with a ``DatetimeIndex``.
    windows : list of int, optional
        Rolling window sizes in periods.  Default is
        ``[21, 63, 126, 189, 252]``, corresponding roughly to 1-month,
        3-month, 6-month, 9-month, and 12-month horizons.
    quantiles : list of float, optional
        Quantile levels to compute for each window.  Default is
        ``[0.1, 0.25, 0.5, 0.75, 0.9]``.

    Returns
    -------
    pd.DataFrame
        Index is the window sizes.  Columns are the quantile labels
        (e.g. ``"10%"``, ``"25%"``, ...) plus ``"current"`` (the most recent
        realized volatility for each window).  All values are annualized
        volatilities.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=504, freq="B")
    >>> rets = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    >>> cone = vol_cone(rets)
    >>> cone.columns.tolist()
    ['10%', '25%', '50%', '75%', '90%', 'current']
    """
    if windows is None:
        windows = [21, 63, 126, 189, 252]
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    ann = _np.sqrt(_annualization_factor(returns))

    records = []
    for w in windows:
        rolling_vol = returns.rolling(window=w, min_periods=w).std() * ann
        rolling_vol = rolling_vol.dropna()

        row = {}
        for q in quantiles:
            label = f"{int(q * 100)}%"
            row[label] = rolling_vol.quantile(q)

        row["current"] = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else _np.nan
        records.append(row)

    df = _pd.DataFrame(records, index=windows)
    df.index.name = "window"
    return df


def vol_term_structure(
    returns: _pd.Series,
    windows: Optional[List[int]] = None,
) -> _pd.Series:
    r"""
    Volatility term structure — current realized volatility at different
    lookback horizons.

    For each window *w*, the function takes the most recent *w* returns and
    computes the annualized standard deviation:

    .. math::

        \sigma^{(w)} = \operatorname{std}(R_{T-w+1}, \ldots, R_{T})
            \cdot \sqrt{N}

    where *T* is the last available observation and *N* is the annualization
    factor.

    Parameters
    ----------
    returns : pd.Series
        A time series of asset returns with a ``DatetimeIndex``.
    windows : list of int, optional
        Lookback window sizes in periods.  Default is
        ``[5, 10, 21, 42, 63, 126, 252]``.

    Returns
    -------
    pd.Series
        Indexed by window sizes, values are annualized volatilities.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=504, freq="B")
    >>> rets = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    >>> ts = vol_term_structure(rets)
    >>> ts.index.tolist()
    [5, 10, 21, 42, 63, 126, 252]
    """
    if windows is None:
        windows = [5, 10, 21, 42, 63, 126, 252]

    ann = _np.sqrt(_annualization_factor(returns))

    vols = {}
    for w in windows:
        recent = returns.iloc[-w:]
        vols[w] = recent.std() * ann

    result = _pd.Series(vols, dtype=float)
    result.index.name = "window"
    result.name = "realized_vol"
    return result


def realized_vol(
    returns: _pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Rolling realized (historical) volatility.

    Computes the standard close-to-close volatility estimator using a rolling
    window of log or simple returns:

    .. math::

        \sigma_{t} = \sqrt{ \frac{1}{n-1}
            \sum_{i=t-n+1}^{t} (R_{i} - \bar{R})^{2} }

    When ``annualize=True`` the result is scaled by :math:`\sqrt{N}` where
    *N* is the number of trading periods per year (252 for daily data).

    Parameters
    ----------
    returns : pd.Series
        A time series of asset returns.
    window : int, optional
        Rolling window size in periods.  Default is 21 (~1 month of
        trading days).
    annualize : bool, optional
        If ``True`` (the default), multiply by :math:`\sqrt{252}` (or the
        inferred annualization factor) to produce an annualized figure.

    Returns
    -------
    pd.Series
        Rolling volatility values.  The first ``window - 1`` entries are
        ``NaN``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    >>> rets = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    >>> rv = realized_vol(rets, window=21)
    >>> rv.dropna().shape[0]
    232
    """
    vol = returns.rolling(window=window, min_periods=window).std()

    if annualize:
        vol = vol * _np.sqrt(_annualization_factor(returns))

    vol.name = "realized_vol"
    return vol


def parkinson_vol(
    high: _pd.Series,
    low: _pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Parkinson (1980) volatility estimator using high-low price ranges.

    The Parkinson estimator exploits the daily high-low range, which contains
    more information about volatility than close-to-close returns alone.
    Under geometric Brownian motion it is approximately **5.2 times** more
    efficient than the classical close-to-close estimator.

    The variance estimate over a rolling window of *n* observations is:

    .. math::

        \hat{\sigma}^{2}_{P}
            = \frac{1}{4\,n\,\ln 2}
              \sum_{i=1}^{n} \bigl[\ln(H_{i} / L_{i})\bigr]^{2}

    Parameters
    ----------
    high : pd.Series
        Daily high prices.
    low : pd.Series
        Daily low prices.
    window : int, optional
        Rolling window size in periods.  Default is 21.
    annualize : bool, optional
        If ``True`` (the default), scale by :math:`\sqrt{N}` where *N* is
        the annualization factor (252 for daily data).

    Returns
    -------
    pd.Series
        Rolling Parkinson volatility estimates.  The first ``window - 1``
        entries are ``NaN``.

    Notes
    -----
    The estimator assumes continuous trading (no overnight jumps) and is
    therefore biased downwards when applied to assets with significant
    close-to-open gaps.

    References
    ----------
    Parkinson, M. (1980). "The Extreme Value Method for Estimating the
    Variance of the Rate of Return". *The Journal of Business*, 53(1), 61-65.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    >>> h = pd.Series(100 + np.cumsum(np.random.normal(0.01, 1, 252)), index=idx)
    >>> l = h - np.abs(np.random.normal(0, 0.5, 252))
    >>> pv = parkinson_vol(h, l, window=21)
    >>> pv.dropna().shape[0]
    232
    """
    log_hl = _np.log(high / low)
    log_hl_sq = log_hl ** 2

    factor = 1.0 / (4.0 * _np.log(2.0))
    variance = log_hl_sq.rolling(window=window, min_periods=window).mean() * factor

    vol = _np.sqrt(variance)

    if annualize:
        vol = vol * _np.sqrt(_annualization_factor(high))

    vol.name = "parkinson_vol"
    return vol


def garman_klass_vol(
    open_price: _pd.Series,
    high: _pd.Series,
    low: _pd.Series,
    close: _pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Garman-Klass (1980) volatility estimator using OHLC data.

    This estimator uses the full set of open-high-low-close prices and is
    approximately **7.4 times** more efficient than the close-to-close
    estimator under geometric Brownian motion.

    The rolling variance estimate over *n* observations is:

    .. math::

        \hat{\sigma}^{2}_{GK}
            = \frac{1}{n} \sum_{i=1}^{n}
              \Bigl[
                  \tfrac{1}{2}\,\bigl[\ln(H_{i}/L_{i})\bigr]^{2}
                  - (2\ln 2 - 1)\,\bigl[\ln(C_{i}/O_{i})\bigr]^{2}
              \Bigr]

    Parameters
    ----------
    open_price : pd.Series
        Daily opening prices.  Named ``open_price`` to avoid shadowing the
        Python built-in ``open``.
    high : pd.Series
        Daily high prices.
    low : pd.Series
        Daily low prices.
    close : pd.Series
        Daily closing prices.
    window : int, optional
        Rolling window size in periods.  Default is 21.
    annualize : bool, optional
        If ``True`` (the default), scale by :math:`\sqrt{N}`.

    Returns
    -------
    pd.Series
        Rolling Garman-Klass volatility estimates.  The first ``window - 1``
        entries are ``NaN``.

    References
    ----------
    Garman, M. B. and Klass, M. J. (1980). "On the Estimation of Security
    Price Volatilities from Historical Data". *The Journal of Business*,
    53(1), 67-78.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> n = 252
    >>> idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    >>> c = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=idx)
    >>> o = c.shift(1).bfill()
    >>> h = pd.concat([o, c], axis=1).max(axis=1) + np.abs(np.random.normal(0, 0.3, n))
    >>> l = pd.concat([o, c], axis=1).min(axis=1) - np.abs(np.random.normal(0, 0.3, n))
    >>> gk = garman_klass_vol(o, h, l, c, window=21)
    >>> gk.dropna().shape[0]
    232
    """
    log_hl = _np.log(high / low)
    log_co = _np.log(close / open_price)

    term1 = 0.5 * log_hl ** 2
    term2 = (2.0 * _np.log(2.0) - 1.0) * log_co ** 2
    gk_daily = term1 - term2

    variance = gk_daily.rolling(window=window, min_periods=window).mean()

    # Floor at zero: the GK daily term can be negative when close-to-open
    # moves dominate high-low range, but the rolling average should not be
    variance = variance.clip(lower=0.0)

    vol = _np.sqrt(variance)

    if annualize:
        vol = vol * _np.sqrt(_annualization_factor(close))

    vol.name = "garman_klass_vol"
    return vol


def yang_zhang_vol(
    open_price: _pd.Series,
    high: _pd.Series,
    low: _pd.Series,
    close: _pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> _pd.Series:
    r"""
    Yang-Zhang (2000) volatility estimator combining overnight, open-to-close,
    and Rogers-Satchell components.

    This is a minimum-variance unbiased estimator that is robust to both
    opening jumps and drift.  It combines three separate variance components:

    **Overnight (open-to-previous-close) variance:**

    .. math::

        \hat{\sigma}^{2}_{o}
            = \frac{1}{n-1} \sum_{i=1}^{n}
              \bigl(\ln(O_{i}/C_{i-1}) - \overline{\ln(O/C_{-1})}\bigr)^{2}

    **Close-to-open variance:**

    .. math::

        \hat{\sigma}^{2}_{c}
            = \frac{1}{n-1} \sum_{i=1}^{n}
              \bigl(\ln(C_{i}/O_{i}) - \overline{\ln(C/O)}\bigr)^{2}

    **Rogers-Satchell variance:**

    .. math::

        \hat{\sigma}^{2}_{rs}
            = \frac{1}{n} \sum_{i=1}^{n}
              \bigl[
                  \ln(H_{i}/O_{i})\,\ln(H_{i}/C_{i})
                  + \ln(L_{i}/O_{i})\,\ln(L_{i}/C_{i})
              \bigr]

    **Yang-Zhang combination:**

    .. math::

        k = \frac{0.34}{1.34 + \frac{n+1}{n-1}}

    .. math::

        \hat{\sigma}^{2}_{YZ}
            = \hat{\sigma}^{2}_{o}
              + k\,\hat{\sigma}^{2}_{c}
              + (1-k)\,\hat{\sigma}^{2}_{rs}

    Parameters
    ----------
    open_price : pd.Series
        Daily opening prices.
    high : pd.Series
        Daily high prices.
    low : pd.Series
        Daily low prices.
    close : pd.Series
        Daily closing prices.
    window : int, optional
        Rolling window size in periods.  Default is 21.
    annualize : bool, optional
        If ``True`` (the default), scale by :math:`\sqrt{N}`.

    Returns
    -------
    pd.Series
        Rolling Yang-Zhang volatility estimates.  The first ``window``
        entries are ``NaN`` (one extra due to the lagged close).

    References
    ----------
    Yang, D. and Zhang, Q. (2000). "Drift Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices". *The Journal of Business*,
    73(3), 477-492.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> n = 252
    >>> idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    >>> c = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=idx)
    >>> o = c.shift(1).bfill()
    >>> h = pd.concat([o, c], axis=1).max(axis=1) + np.abs(np.random.normal(0, 0.3, n))
    >>> l = pd.concat([o, c], axis=1).min(axis=1) - np.abs(np.random.normal(0, 0.3, n))
    >>> yz = yang_zhang_vol(o, h, l, c, window=21)
    >>> yz.dropna().shape[0] > 200
    True
    """
    # Lagged close (previous day's close)
    close_prev = close.shift(1)

    # Log return components
    log_oc_prev = _np.log(open_price / close_prev)  # overnight
    log_co = _np.log(close / open_price)             # close-to-open (intraday)
    log_ho = _np.log(high / open_price)
    log_hc = _np.log(high / close)
    log_lo = _np.log(low / open_price)
    log_lc = _np.log(low / close)

    # Rogers-Satchell daily component
    rs_daily = log_ho * log_hc + log_lo * log_lc

    # Weighting constant
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Rolling overnight variance: (1/(n-1)) * Σ(x_i - mean)^2
    # This is simply the rolling variance (which uses ddof=1 by default)
    sigma2_o = log_oc_prev.rolling(window=window, min_periods=window).var()

    # Rolling close-to-open variance
    sigma2_c = log_co.rolling(window=window, min_periods=window).var()

    # Rolling Rogers-Satchell variance (uses population mean, i.e. just the mean)
    sigma2_rs = rs_daily.rolling(window=window, min_periods=window).mean()

    # Yang-Zhang combined variance
    sigma2_yz = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs

    vol = _np.sqrt(sigma2_yz)

    if annualize:
        vol = vol * _np.sqrt(_annualization_factor(close))

    vol.name = "yang_zhang_vol"
    return vol
