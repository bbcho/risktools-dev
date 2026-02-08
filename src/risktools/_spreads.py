# Commodity Spreads Analytics

from typing import Union, Optional

import pandas as _pd
import numpy as _np

__all__ = [
    "crack_spread",
    "spark_spread",
    "crush_spread",
    "convenience_yield",
    "optimal_hedge_ratio",
    "calendar_spread",
    "fly_spread",
]

_GAL_PER_BBL = 42


def _align_series(*series, names=None):
    """Align multiple pd.Series on their DatetimeIndex using inner join.

    Parameters
    ----------
    *series : pd.Series
        Variable number of pandas Series to align.
    names : list of str, optional
        Names for each series, used in error messages.

    Returns
    -------
    list of pd.Series
        Aligned series with a common DatetimeIndex.

    Raises
    ------
    ValueError
        If any input is not a pd.Series, lacks a DatetimeIndex, or if there
        are no overlapping dates after alignment.
    """
    if names is None:
        names = [f"series_{i}" for i in range(len(series))]

    for s, name in zip(series, names):
        if not isinstance(s, _pd.Series):
            raise TypeError(f"{name} must be a pandas Series, got {type(s).__name__}")
        if not isinstance(s.index, _pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex")
        if not _np.issubdtype(s.dtype, _np.number):
            raise ValueError(f"{name} must contain numeric data")

    # inner join on index
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)

    if len(idx) == 0:
        raise ValueError("No overlapping dates found between the input series")

    return [s.loc[idx] for s in series]


def crack_spread(
    crude: _pd.Series,
    gasoline: _pd.Series,
    heating_oil: Optional[_pd.Series] = None,
    ratio: str = "3:2:1",
) -> _pd.Series:
    r"""
    Compute crack spread from crude oil and refined product prices.

    The crack spread represents the refining margin — the difference between
    the value of refined petroleum products and the cost of crude oil. Various
    crack spread ratios approximate typical refinery yield profiles.

    Parameters
    ----------
    crude : pd.Series
        Crude oil prices in $/bbl with a DatetimeIndex.
    gasoline : pd.Series
        Gasoline prices in $/gal with a DatetimeIndex.
    heating_oil : pd.Series, optional
        Heating oil (or diesel / ULSD) prices in $/gal with a DatetimeIndex.
        Required for all ratios except ``"1:1"``.
    ratio : str, default ``"3:2:1"``
        Crack spread ratio. Supported values:

        - ``"3:2:1"``: 2 barrels gasoline + 1 barrel heating oil from 3 barrels crude
        - ``"5:3:2"``: 3 barrels gasoline + 2 barrels heating oil from 5 barrels crude
        - ``"2:1:1"``: 1 barrel gasoline + 1 barrel heating oil from 2 barrels crude
        - ``"1:1"``: simple crack — 1 barrel gasoline from 1 barrel crude

    Returns
    -------
    pd.Series
        Crack spread values in $/bbl with a DatetimeIndex.

    Notes
    -----
    Product prices in $/gal are converted to $/bbl by multiplying by 42
    (gallons per barrel).

    For the 3:2:1 crack spread, the formula is:

    .. math:: \text{crack}_{3:2:1} = \frac{2 \times P_g \times 42 + 1 \times P_h \times 42 - 3 \times P_c}{3}

    where :math:`P_g` is the gasoline price ($/gal), :math:`P_h` is the
    heating oil price ($/gal), and :math:`P_c` is the crude oil price ($/bbl).

    For the simple 1:1 crack spread:

    .. math:: \text{crack}_{1:1} = P_g \times 42 - P_c

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> crude = pd.Series([70.0, 71.0, 72.0], index=dates)
    >>> gasoline = pd.Series([2.50, 2.55, 2.60], index=dates)
    >>> ho = pd.Series([2.80, 2.85, 2.90], index=dates)
    >>> crack_spread(crude, gasoline, ho, ratio="3:2:1")
    """
    _valid_ratios = {"3:2:1", "5:3:2", "2:1:1", "1:1"}
    if ratio not in _valid_ratios:
        raise ValueError(f"ratio must be one of {_valid_ratios}, got '{ratio}'")

    if ratio == "1:1":
        crude, gasoline = _align_series(
            crude, gasoline, names=["crude", "gasoline"]
        )
        result = gasoline * _GAL_PER_BBL - crude
    else:
        if heating_oil is None:
            raise ValueError(
                f"heating_oil is required for ratio '{ratio}'"
            )
        crude, gasoline, heating_oil = _align_series(
            crude, gasoline, heating_oil,
            names=["crude", "gasoline", "heating_oil"],
        )

        if ratio == "3:2:1":
            result = (
                2 * gasoline * _GAL_PER_BBL
                + 1 * heating_oil * _GAL_PER_BBL
                - 3 * crude
            ) / 3
        elif ratio == "5:3:2":
            result = (
                3 * gasoline * _GAL_PER_BBL
                + 2 * heating_oil * _GAL_PER_BBL
                - 5 * crude
            ) / 5
        elif ratio == "2:1:1":
            result = (
                1 * gasoline * _GAL_PER_BBL
                + 1 * heating_oil * _GAL_PER_BBL
                - 2 * crude
            ) / 2

    result.name = f"crack_spread_{ratio}"
    return result


def spark_spread(
    power: _pd.Series,
    natural_gas: _pd.Series,
    heat_rate: float = 7.0,
) -> _pd.Series:
    r"""
    Compute spark spread from electricity and natural gas prices.

    The spark spread measures the theoretical gross margin of a gas-fired
    power plant — the difference between the revenue from selling electricity
    and the cost of the natural gas fuel needed to generate it.

    Parameters
    ----------
    power : pd.Series
        Electricity prices in $/MWh with a DatetimeIndex.
    natural_gas : pd.Series
        Natural gas prices in $/MMBtu with a DatetimeIndex.
    heat_rate : float, default 7.0
        Heat rate of the generating unit in MMBtu/MWh. A lower heat rate
        indicates a more efficient plant. Typical values:

        - ~7.0 for an efficient combined-cycle gas turbine (CCGT)
        - ~10.0 for a simple-cycle gas turbine (peaker)

    Returns
    -------
    pd.Series
        Spark spread values in $/MWh with a DatetimeIndex.

    Notes
    -----
    The spark spread is defined as:

    .. math:: \text{spark} = P_{\text{power}} - P_{\text{gas}} \times HR

    where :math:`P_{\text{power}}` is the electricity price ($/MWh),
    :math:`P_{\text{gas}}` is the natural gas price ($/MMBtu), and
    :math:`HR` is the heat rate (MMBtu/MWh).

    A positive spark spread indicates that it is profitable to run the plant
    (before accounting for variable O&M, emissions costs, etc.).

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> power = pd.Series([45.0, 50.0, 48.0], index=dates)
    >>> gas = pd.Series([3.50, 3.60, 3.55], index=dates)
    >>> spark_spread(power, gas, heat_rate=7.0)
    """
    if heat_rate <= 0:
        raise ValueError(f"heat_rate must be positive, got {heat_rate}")

    power, natural_gas = _align_series(
        power, natural_gas, names=["power", "natural_gas"]
    )

    result = power - natural_gas * heat_rate
    result.name = "spark_spread"
    return result


def crush_spread(
    soybeans: _pd.Series,
    soybean_meal: _pd.Series,
    soybean_oil: _pd.Series,
) -> _pd.Series:
    r"""
    Compute the soybean crush spread (board crush margin).

    The crush spread represents the gross processing margin for crushing
    soybeans into its two primary products — soybean meal and soybean oil.

    Parameters
    ----------
    soybeans : pd.Series
        Soybean prices in cents/bushel with a DatetimeIndex.
    soybean_meal : pd.Series
        Soybean meal prices in $/short ton with a DatetimeIndex.
    soybean_oil : pd.Series
        Soybean oil prices in cents/lb with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Crush spread values in $/bushel with a DatetimeIndex.

    Notes
    -----
    The standard CBOT board crush assumes that one bushel of soybeans
    yields 48 lbs of soybean meal and 11 lbs of soybean oil. The crush
    margin is:

    .. math:: \text{crush} = \underbrace{P_m \times \frac{48}{2000}}_{\text{meal value}} + \underbrace{\frac{P_o \times 11}{100}}_{\text{oil value}} - \underbrace{\frac{P_s}{100}}_{\text{bean cost}}

    where:

    - :math:`P_m` is the soybean meal price ($/short ton)
    - :math:`P_o` is the soybean oil price (cents/lb)
    - :math:`P_s` is the soybean price (cents/bushel)

    The meal conversion factor is :math:`48 / 2000 = 0.024` (48 lbs of meal
    per bushel, 2000 lbs per short ton). The oil conversion divides by 100
    to convert cents to dollars.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> beans = pd.Series([1300.0, 1310.0, 1305.0], index=dates)
    >>> meal = pd.Series([380.0, 385.0, 382.0], index=dates)
    >>> oil = pd.Series([55.0, 56.0, 55.5], index=dates)
    >>> crush_spread(beans, meal, oil)
    """
    soybeans, soybean_meal, soybean_oil = _align_series(
        soybeans, soybean_meal, soybean_oil,
        names=["soybeans", "soybean_meal", "soybean_oil"],
    )

    meal_value = soybean_meal * 0.024       # $/ton * (48 lbs / 2000 lbs/ton)
    oil_value = soybean_oil * 11 / 100      # cents/lb * 11 lbs / 100 -> $
    bean_cost = soybeans / 100              # cents/bu -> $/bu

    result = meal_value + oil_value - bean_cost
    result.name = "crush_spread"
    return result


def convenience_yield(
    spot: Union[float, _pd.Series],
    futures: Union[float, _pd.Series],
    r: float,
    T: float,
) -> Union[float, _pd.Series]:
    r"""
    Extract the implied convenience yield from spot and futures prices.

    Uses the cost-of-carry model to back out the convenience yield that
    reconciles observed spot and futures prices. The convenience yield
    represents the non-monetary benefit of holding the physical commodity
    (e.g., ability to meet unexpected demand, keep a production process running).

    Parameters
    ----------
    spot : float or pd.Series
        Spot price(s) of the commodity.
    futures : float or pd.Series
        Futures price(s) of the commodity.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    T : float
        Time to maturity of the futures contract in years.

    Returns
    -------
    float or pd.Series
        Annualized implied convenience yield.

    Notes
    -----
    Under the cost-of-carry model with continuous compounding, the
    relationship between spot and futures prices is:

    .. math:: F = S \, e^{(r - y) T}

    where :math:`F` is the futures price, :math:`S` is the spot price,
    :math:`r` is the risk-free rate, :math:`y` is the convenience yield,
    and :math:`T` is the time to maturity.

    Solving for the convenience yield:

    .. math:: y = r - \frac{1}{T} \ln\!\left(\frac{F}{S}\right)

    A positive convenience yield (:math:`y > 0`) indicates the market is in
    backwardation — futures trade below the full cost-of-carry price — which
    is common when inventories are low.

    Examples
    --------
    >>> convenience_yield(spot=100.0, futures=102.0, r=0.05, T=0.5)
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    if isinstance(spot, _pd.Series) and isinstance(futures, _pd.Series):
        spot, futures = _align_series(
            spot, futures, names=["spot", "futures"]
        )

    y = r - _np.log(futures / spot) / T

    if isinstance(y, _pd.Series):
        y.name = "convenience_yield"

    return y


def optimal_hedge_ratio(
    spot_returns: _pd.Series,
    futures_returns: _pd.Series,
    method: str = "ols",
) -> dict:
    r"""
    Compute the optimal (minimum variance) hedge ratio.

    Estimates the proportion of a spot position that should be hedged using
    futures contracts to minimize the variance of the hedged portfolio.

    Parameters
    ----------
    spot_returns : pd.Series
        Returns (or price changes) of the spot position with a DatetimeIndex.
    futures_returns : pd.Series
        Returns (or price changes) of the futures contract with a DatetimeIndex.
    method : str, default ``"ols"``
        Estimation method:

        - ``"ols"``: OLS regression of spot returns on futures returns. The
          slope coefficient is the hedge ratio.
        - ``"min_variance"``: Direct calculation via
          :math:`\text{Cov}(\Delta S, \Delta F) / \text{Var}(\Delta F)`.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"hedge_ratio"`` : float — the optimal :math:`h^*`
        - ``"r_squared"`` : float — coefficient of determination (hedge effectiveness)
        - ``"residual_variance"`` : float — variance of the hedged portfolio residuals

    Notes
    -----
    The minimum-variance hedge ratio minimizes the variance of the hedged
    portfolio return :math:`R_h = \Delta S - h \, \Delta F`. Taking the
    first-order condition:

    .. math:: h^* = \frac{\text{Cov}(\Delta S, \Delta F)}{\text{Var}(\Delta F)}

    This is algebraically equivalent to the OLS slope from regressing
    :math:`\Delta S` on :math:`\Delta F`:

    .. math:: \Delta S_t = \alpha + h^* \, \Delta F_t + \varepsilon_t

    The :math:`R^2` of this regression is the *hedge effectiveness* — the
    proportion of spot return variance eliminated by hedging:

    .. math:: R^2 = \frac{h^{*2} \, \text{Var}(\Delta F)}{\text{Var}(\Delta S)}

    References
    ----------
    Hull, John C. *Options, Futures, and Other Derivatives*. Chapter 3,
    "Hedging Strategies Using Futures."

    Ederington, Louis H. "The Hedging Performance of the New Futures Markets."
    *Journal of Finance*, 34(1), 1979.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2024-01-01", periods=100, freq="B")
    >>> spot_ret = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
    >>> fut_ret = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
    >>> optimal_hedge_ratio(spot_ret, fut_ret)
    """
    _valid_methods = {"ols", "min_variance"}
    if method not in _valid_methods:
        raise ValueError(f"method must be one of {_valid_methods}, got '{method}'")

    spot_returns, futures_returns = _align_series(
        spot_returns, futures_returns,
        names=["spot_returns", "futures_returns"],
    )

    # Drop any remaining NaN pairs
    mask = spot_returns.notna() & futures_returns.notna()
    ds = spot_returns[mask].values
    df = futures_returns[mask].values

    if len(ds) < 2:
        raise ValueError("Need at least 2 overlapping non-NaN observations")

    if method == "ols":
        # OLS: ΔS = α + h* ΔF + ε
        df_with_const = _np.column_stack([_np.ones(len(df)), df])
        # Solve via least squares
        coeffs, residuals, _, _ = _np.linalg.lstsq(df_with_const, ds, rcond=None)
        h_star = coeffs[1]

        # Fitted values and R²
        fitted = df_with_const @ coeffs
        ss_res = _np.sum((ds - fitted) ** 2)
        ss_tot = _np.sum((ds - _np.mean(ds)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_var = ss_res / (len(ds) - 2)

    else:  # min_variance
        cov_sf = _np.cov(ds, df, ddof=1)[0, 1]
        var_f = _np.var(df, ddof=1)
        if var_f == 0:
            raise ValueError("futures_returns has zero variance; cannot compute hedge ratio")
        h_star = cov_sf / var_f

        # Compute R² and residual variance for consistency
        hedged = ds - h_star * df
        ss_res = _np.sum((hedged - _np.mean(hedged)) ** 2)
        ss_tot = _np.sum((ds - _np.mean(ds)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_var = _np.var(hedged, ddof=1)

    return {
        "hedge_ratio": float(h_star),
        "r_squared": float(r_squared),
        "residual_variance": float(residual_var),
    }


def calendar_spread(
    front: _pd.Series,
    back: _pd.Series,
) -> _pd.Series:
    r"""
    Compute the calendar (time) spread between two contract months.

    The calendar spread is the price difference between a nearer-dated
    (front) contract and a further-dated (back) contract. It provides
    insight into the term structure of commodity futures.

    Parameters
    ----------
    front : pd.Series
        Front-month (nearby) contract prices with a DatetimeIndex.
    back : pd.Series
        Back-month (deferred) contract prices with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Calendar spread values with a DatetimeIndex. Positive values
        indicate backwardation (front > back); negative values indicate
        contango (front < back).

    Notes
    -----
    The calendar spread is defined simply as:

    .. math:: \text{spread} = F_{\text{front}} - F_{\text{back}}

    - **Backwardation** (:math:`\text{spread} > 0`): near-term prices exceed
      deferred prices, often indicating tight current supply.
    - **Contango** (:math:`\text{spread} < 0`): deferred prices exceed
      near-term prices, the normal condition reflecting storage costs.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> front = pd.Series([75.0, 74.5, 76.0], index=dates)
    >>> back = pd.Series([73.0, 73.5, 74.0], index=dates)
    >>> calendar_spread(front, back)
    """
    front, back = _align_series(front, back, names=["front", "back"])

    result = front - back
    result.name = "calendar_spread"
    return result


def fly_spread(
    front: _pd.Series,
    middle: _pd.Series,
    back: _pd.Series,
) -> _pd.Series:
    r"""
    Compute the butterfly spread across three contract months.

    The butterfly spread measures the curvature of the futures term structure.
    It is constructed by going long the front and back contracts and short
    twice the middle contract.

    Parameters
    ----------
    front : pd.Series
        Front-month contract prices with a DatetimeIndex.
    middle : pd.Series
        Middle-month contract prices with a DatetimeIndex.
    back : pd.Series
        Back-month contract prices with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Butterfly spread values with a DatetimeIndex.

    Notes
    -----
    The butterfly spread is defined as:

    .. math:: \text{fly} = F_{\text{front}} - 2 \, F_{\text{middle}} + F_{\text{back}}

    This can be decomposed into two calendar spreads:

    .. math:: \text{fly} = (F_{\text{front}} - F_{\text{middle}}) - (F_{\text{middle}} - F_{\text{back}})

    A positive butterfly indicates that the front calendar spread is wider
    than the back calendar spread (convex term structure). The butterfly is
    commonly used to trade relative value along the forward curve without
    taking a directional view.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2024-01-01", periods=3, freq="B")
    >>> front = pd.Series([75.0, 74.5, 76.0], index=dates)
    >>> middle = pd.Series([74.0, 73.5, 75.0], index=dates)
    >>> back = pd.Series([73.5, 73.0, 74.5], index=dates)
    >>> fly_spread(front, middle, back)
    """
    front, middle, back = _align_series(
        front, middle, back, names=["front", "middle", "back"]
    )

    result = front - 2 * middle + back
    result.name = "fly_spread"
    return result
