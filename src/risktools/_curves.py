# Forward Curve Analytics

from typing import Union

import pandas as _pd
import numpy as _np

__all__ = [
    "curve_calendar_spread",
    "curve_fly",
    "curve_shape",
    "basis",
    "roll_yield",
    "term_structure_slope",
    "curve_seasonality",
]


def curve_calendar_spread(
    curve: Union[_pd.DataFrame, _pd.Series],
    front_idx: int = 0,
    back_idx: int = 1,
) -> _pd.Series:
    r"""
    Compute calendar spread from a forward curve.

    The calendar spread measures the price difference between two contract
    months along a forward curve.  A positive spread indicates the front
    contract trades above the back contract (backwardation).

    **DataFrame input** (time series of curves):

    .. math:: \text{spread}_t = F_{t,\,\text{front}} - F_{t,\,\text{back}}

    where :math:`F_{t,\,j}` is the futures price observed on date *t* for
    contract month *j*.

    **Series input** (single curve snapshot):

    .. math:: \text{spread}_i = P_i - P_{i+1}

    i.e. the first difference of adjacent contract prices.

    Parameters
    ----------
    curve : pd.DataFrame or pd.Series
        If DataFrame: rows are observation dates (DatetimeIndex), columns are
        contract months.  ``front_idx`` and ``back_idx`` select columns by
        position.

        If Series: values are prices indexed by contract/delivery month.
        The function returns the first difference (adjacent month spread);
        ``front_idx`` and ``back_idx`` are ignored.
    front_idx : int, optional
        Column index of the front (nearby) contract when *curve* is a
        DataFrame.  Default 0.
    back_idx : int, optional
        Column index of the back (deferred) contract when *curve* is a
        DataFrame.  Default 1.

    Returns
    -------
    pd.Series
        Calendar spread values.  Positive values indicate backwardation
        (front > back).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=4, freq="ME")
    >>> curve = pd.Series([70.0, 69.5, 68.0, 67.0], index=idx)
    >>> curve_calendar_spread(curve)  # first differences
    2024-02-29    0.5
    2024-03-31    1.5
    2024-04-30    1.0
    Freq: ME, dtype: float64
    """
    if isinstance(curve, _pd.DataFrame):
        if curve.shape[1] < 2:
            raise ValueError(
                "DataFrame must have at least 2 columns to compute a calendar spread"
            )
        front_col = curve.iloc[:, front_idx]
        back_col = curve.iloc[:, back_idx]
        spread = front_col - back_col
        spread.name = "calendar_spread"
        return spread

    if isinstance(curve, _pd.Series):
        spread = -curve.diff()
        # diff() gives curve[i] - curve[i-1]; we want curve[i] - curve[i+1],
        # so shift by one period forward, or equivalently negate diff and drop
        # the first NaN.
        spread = spread.iloc[1:]
        # Re-index: each spread value corresponds to the *front* contract of
        # the pair.  diff() already assigned the later index; shift back.
        spread.index = curve.index[:-1]
        # Actually: diff gives P[i] - P[i-1]. Negated gives P[i-1] - P[i].
        # After dropping first NaN and re-indexing to front, that is exactly
        # front - back for consecutive pairs.
        spread.name = "calendar_spread"
        return spread

    raise TypeError("curve must be a pandas DataFrame or Series")


def curve_fly(
    curve: _pd.DataFrame,
    front_idx: int = 0,
    middle_idx: int = 1,
    back_idx: int = 2,
) -> _pd.Series:
    r"""
    Compute butterfly spread from a forward curve.

    The butterfly spread captures the curvature (convexity) of the forward
    curve around a middle contract month:

    .. math:: \text{fly}_t = F_{t,\,\text{front}} - 2\,F_{t,\,\text{middle}} + F_{t,\,\text{back}}

    A positive butterfly indicates the middle contract is cheap relative
    to its neighbours; a negative butterfly indicates a rich middle
    contract.

    Parameters
    ----------
    curve : pd.DataFrame
        Rows are observation dates (DatetimeIndex), columns are contract
        months.  Must have at least 3 columns.
    front_idx : int, optional
        Column index of the front leg.  Default 0.
    middle_idx : int, optional
        Column index of the middle (body) leg.  Default 1.
    back_idx : int, optional
        Column index of the back leg.  Default 2.

    Returns
    -------
    pd.Series
        Butterfly spread values.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> df = pd.DataFrame(
    ...     {"M1": [70, 71, 69], "M2": [69, 70, 68], "M3": [67, 68, 66]},
    ...     index=dates,
    ... )
    >>> curve_fly(df)
    2024-01-01   -1.0
    2024-01-02   -1.0
    2024-01-03   -1.0
    Freq: B, Name: butterfly, dtype: float64
    """
    if not isinstance(curve, _pd.DataFrame):
        raise TypeError("curve must be a pandas DataFrame")
    if curve.shape[1] < 3:
        raise ValueError(
            "DataFrame must have at least 3 columns to compute a butterfly spread"
        )

    front = curve.iloc[:, front_idx]
    middle = curve.iloc[:, middle_idx]
    back = curve.iloc[:, back_idx]

    fly = front - 2 * middle + back
    fly.name = "butterfly"
    return fly


def curve_shape(
    curve: Union[_pd.Series, _pd.DataFrame],
) -> Union[dict, _pd.DataFrame]:
    r"""
    Analyse forward curve shape -- contango, backwardation, or mixed.

    For a single curve snapshot (Series), the function fits linear and
    quadratic models of price on time-to-delivery and classifies the
    curve shape:

    **Linear model:**

    .. math:: P_i = \alpha + \beta \, \tau_i + \varepsilon_i

    where :math:`\tau_i` is the time (in years) from the first contract
    to contract *i*.  The slope :math:`\beta` is positive for contango
    and negative for backwardation.

    **Quadratic model:**

    .. math:: P_i = a + b\,\tau_i + c\,\tau_i^2 + \varepsilon_i

    The curvature coefficient :math:`c` measures the degree of convexity
    in the term structure.

    Parameters
    ----------
    curve : pd.Series or pd.DataFrame
        **Series**: prices indexed by delivery/expiry dates (a single
        term-structure snapshot).

        **DataFrame**: each row is a curve snapshot (observation date as
        index), columns are contract months.

    Returns
    -------
    dict or pd.DataFrame
        **Series input** returns a dict with keys:

        - ``"shape"`` : str -- ``"contango"``, ``"backwardation"``, or
          ``"mixed"``
        - ``"slope"`` : float -- linear regression slope
          (:math:`\beta`, price units per year)
        - ``"curvature"`` : float -- quadratic coefficient :math:`c`
        - ``"max_backwardation"`` : float -- largest negative
          month-to-month price change
        - ``"max_contango"`` : float -- largest positive month-to-month
          price change

        **DataFrame input** returns a DataFrame with columns
        ``shape``, ``slope``, ``curvature`` indexed by observation date.

    Examples
    --------
    >>> import pandas as pd
    >>> months = pd.date_range("2024-01-01", periods=6, freq="ME")
    >>> prices = pd.Series([70, 71, 72, 73, 74, 75], index=months, dtype=float)
    >>> result = curve_shape(prices)
    >>> result["shape"]
    'contango'
    """
    if isinstance(curve, _pd.Series):
        return _curve_shape_single(curve)

    if isinstance(curve, _pd.DataFrame):
        rows = []
        for date, row in curve.iterrows():
            series = row.dropna()
            if len(series) < 2:
                rows.append(
                    {"shape": _np.nan, "slope": _np.nan, "curvature": _np.nan}
                )
                continue
            # Build a Series with a "virtual" DatetimeIndex if columns are not
            # already datetime.
            if isinstance(series.index, _pd.DatetimeIndex):
                snap = series
            else:
                # Assume columns are ordered contract months; create a
                # monthly date range as a proxy for delivery dates.
                snap = _pd.Series(
                    series.values,
                    index=_pd.date_range("2000-01-01", periods=len(series), freq="ME"),
                )
            info = _curve_shape_single(snap)
            rows.append(
                {
                    "shape": info["shape"],
                    "slope": info["slope"],
                    "curvature": info["curvature"],
                }
            )
        return _pd.DataFrame(rows, index=curve.index)

    raise TypeError("curve must be a pandas Series or DataFrame")


def _curve_shape_single(curve: _pd.Series) -> dict:
    """Analyse shape of a single curve snapshot (Series)."""
    prices = curve.dropna()
    n = len(prices)

    if n == 0:
        return {
            "shape": _np.nan,
            "slope": _np.nan,
            "curvature": _np.nan,
            "max_backwardation": _np.nan,
            "max_contango": _np.nan,
        }

    if n == 1:
        return {
            "shape": "flat",
            "slope": 0.0,
            "curvature": 0.0,
            "max_backwardation": 0.0,
            "max_contango": 0.0,
        }

    # Time vector in years from first delivery date
    if isinstance(prices.index, _pd.DatetimeIndex):
        days = (prices.index - prices.index[0]).days.astype(float)
        tau = days / 365.25
    else:
        tau = _np.arange(n, dtype=float)

    # Linear fit: P = alpha + beta * tau
    coeffs_lin = _np.polyfit(tau, prices.values, 1)
    slope = coeffs_lin[0]  # beta (price units per year)
    intercept = coeffs_lin[1]

    # Quadratic fit: P = a + b*tau + c*tau^2
    if n >= 3:
        coeffs_quad = _np.polyfit(tau, prices.values, 2)
        curvature = coeffs_quad[0]  # coefficient of tau^2
    else:
        curvature = 0.0

    # Month-to-month differences
    diffs = _np.diff(prices.values)
    max_contango = float(diffs.max()) if len(diffs) > 0 else 0.0
    max_backwardation = float(diffs.min()) if len(diffs) > 0 else 0.0

    # Classification
    if _np.all(diffs >= 0):
        if _np.all(diffs == 0):
            shape = "flat"
        else:
            shape = "contango"
    elif _np.all(diffs <= 0):
        shape = "backwardation"
    else:
        shape = "mixed"

    return {
        "shape": shape,
        "slope": float(slope),
        "curvature": float(curvature),
        "max_backwardation": max_backwardation,
        "max_contango": max_contango,
    }


def basis(
    spot: _pd.Series,
    front_futures: _pd.Series,
) -> _pd.Series:
    r"""
    Compute the basis (spot minus futures).

    .. math:: \text{basis}_t = S_t - F_t

    where :math:`S_t` is the spot price and :math:`F_t` is the front-month
    futures price.  A positive basis indicates backwardation (spot premium);
    a negative basis indicates contango (futures premium).

    The two input Series are aligned on their index using an inner join
    so that only common dates are retained.

    Parameters
    ----------
    spot : pd.Series
        Spot prices.
    front_futures : pd.Series
        Front-month futures prices.

    Returns
    -------
    pd.Series
        Basis values (positive = backwardation / spot premium).

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> spot = pd.Series([70.0, 71.0, 69.5], index=dates)
    >>> fut = pd.Series([69.5, 70.5, 69.0], index=dates)
    >>> basis(spot, fut)
    2024-01-01    0.5
    2024-01-02    0.5
    2024-01-03    0.5
    Freq: B, Name: basis, dtype: float64
    """
    if not isinstance(spot, _pd.Series):
        raise TypeError("spot must be a pandas Series")
    if not isinstance(front_futures, _pd.Series):
        raise TypeError("front_futures must be a pandas Series")

    # Align on common index dates
    spot_aligned, fut_aligned = spot.align(front_futures, join="inner")

    result = spot_aligned - fut_aligned
    result.name = "basis"
    return result


def roll_yield(
    front_futures: _pd.Series,
    next_futures: _pd.Series,
) -> _pd.Series:
    r"""
    Compute the roll yield from front and next (deferred) contract prices.

    The roll yield captures the return earned (or paid) when a futures
    position is rolled from the expiring front contract into the next
    contract:

    .. math:: y_t^{\text{roll}} = \frac{F_{t,1} - F_{t,2}}{F_{t,2}}

    where :math:`F_{t,1}` is the front-month price and :math:`F_{t,2}` is
    the next-month price.  A positive roll yield indicates the front
    contract trades at a premium (backwardation), meaning a long position
    earns positive carry when rolling.

    To annualise for monthly contracts, multiply by 12.

    The two input Series are aligned on their index using an inner join.

    Parameters
    ----------
    front_futures : pd.Series
        Front-month futures prices.
    next_futures : pd.Series
        Next-month (deferred) futures prices.

    Returns
    -------
    pd.Series
        Roll yield values (dimensionless).

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> f1 = pd.Series([70.0, 71.0, 69.0], index=dates)
    >>> f2 = pd.Series([69.0, 70.0, 68.5], index=dates)
    >>> roll_yield(f1, f2)
    2024-01-01    0.014493
    2024-01-02    0.014286
    2024-01-03    0.007299
    Freq: B, Name: roll_yield, dtype: float64
    """
    if not isinstance(front_futures, _pd.Series):
        raise TypeError("front_futures must be a pandas Series")
    if not isinstance(next_futures, _pd.Series):
        raise TypeError("next_futures must be a pandas Series")

    front_aligned, next_aligned = front_futures.align(next_futures, join="inner")

    result = (front_aligned - next_aligned) / next_aligned
    result.name = "roll_yield"
    return result


def term_structure_slope(
    curve: _pd.Series,
    method: str = "linear",
) -> dict:
    r"""
    Estimate the slope of a forward / futures term structure.

    **Linear method** (``method="linear"``):

    .. math:: P_i = \alpha + \beta\,\tau_i + \varepsilon_i

    **Log method** (``method="log"``):

    .. math:: \ln P_i = \alpha + \beta\,\tau_i + \varepsilon_i

    where :math:`\tau_i` is the time in years from the first delivery date
    to delivery date *i*.  The slope :math:`\beta` represents the average
    rate of change per year along the curve.

    Standard error is computed from the OLS residuals:

    .. math::
        \text{SE}(\hat\beta) = \sqrt{\frac{\sum \hat\varepsilon_i^2}
        {(n-2)\,\sum(\tau_i - \bar\tau)^2}}

    Parameters
    ----------
    curve : pd.Series
        Prices indexed by delivery/expiry dates (DatetimeIndex).
    method : str, optional
        ``"linear"`` (default) or ``"log"``.

    Returns
    -------
    dict
        - ``"slope"`` : float -- :math:`\beta` (price units per year for
          linear, log-price per year for log)
        - ``"intercept"`` : float -- :math:`\alpha`
        - ``"r_squared"`` : float -- :math:`R^2` of the fit
        - ``"std_err"`` : float -- standard error of the slope estimate

    Examples
    --------
    >>> import pandas as pd
    >>> months = pd.date_range("2024-01-01", periods=6, freq="ME")
    >>> prices = pd.Series([70.0, 71.0, 72.0, 73.0, 74.0, 75.0], index=months)
    >>> result = term_structure_slope(prices)
    >>> result["slope"] > 0
    True
    """
    if not isinstance(curve, _pd.Series):
        raise TypeError("curve must be a pandas Series")
    if method not in ("linear", "log"):
        raise ValueError('method must be "linear" or "log"')

    prices = curve.dropna()
    n = len(prices)

    if n < 2:
        raise ValueError("curve must have at least 2 non-null observations")

    # Build time vector in years
    if isinstance(prices.index, _pd.DatetimeIndex):
        days = (prices.index - prices.index[0]).days.astype(float)
        tau = days / 365.25
    else:
        tau = _np.arange(n, dtype=float)

    y = prices.values.astype(float)
    if method == "log":
        if _np.any(y <= 0):
            raise ValueError("All prices must be positive for log method")
        y = _np.log(y)

    # OLS via numpy polyfit (degree 1)
    coeffs, cov = _np.polyfit(tau, y, 1, cov=True)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R-squared
    y_hat = _np.polyval(coeffs, tau)
    ss_res = float(_np.sum((y - y_hat) ** 2))
    ss_tot = float(_np.sum((y - _np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Standard error of slope from covariance matrix
    std_err = float(_np.sqrt(cov[0, 0]))

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "std_err": std_err,
    }


def curve_seasonality(
    prices: _pd.Series,
    freq: str = "M",
) -> _pd.DataFrame:
    r"""
    Extract the seasonal pattern from a price series.

    The seasonal factor for each period :math:`k` is computed as:

    .. math:: s_k = \bar{P}_k - \bar{P}

    where :math:`\bar{P}_k` is the mean price in period *k* and
    :math:`\bar{P}` is the overall mean price.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex.
    freq : str, optional
        Grouping frequency:

        - ``"M"`` -- monthly (periods 1--12), default
        - ``"W"`` -- weekly  (periods 1--52)
        - ``"Q"`` -- quarterly (periods 1--4)

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``"mean"`` -- average price in each period
        - ``"std"`` -- standard deviation of prices in each period
        - ``"count"`` -- number of observations in each period
        - ``"seasonal_factor"`` -- deviation of period mean from overall
          mean (:math:`s_k`)

        Indexed by period number (1--12 for monthly, 1--52 for weekly,
        1--4 for quarterly).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2020-01-01", periods=36, freq="ME")
    >>> prices = pd.Series(np.random.default_rng(42).normal(100, 5, 36), index=idx)
    >>> seas = curve_seasonality(prices, freq="M")
    >>> seas.index.tolist()
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """
    if not isinstance(prices, _pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not isinstance(prices.index, _pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if freq not in ("M", "W", "Q"):
        raise ValueError('freq must be "M", "W", or "Q"')

    prices = prices.dropna()

    if freq == "M":
        grouper = prices.index.month
        period_name = "month"
    elif freq == "W":
        grouper = prices.index.isocalendar().week.values.astype(int)
        period_name = "week"
    else:  # "Q"
        grouper = prices.index.quarter
        period_name = "quarter"

    grouped = prices.groupby(grouper)
    overall_mean = float(prices.mean())

    result = _pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "count": grouped.count(),
        }
    )
    result["seasonal_factor"] = result["mean"] - overall_mean
    result.index.name = period_name
    return result
