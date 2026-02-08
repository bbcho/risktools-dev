# Options Pricing Functions

import numpy as _np
from scipy.stats import norm as _norm
from typing import Union, Optional

__all__ = [
    "black_scholes",
    "black_scholes_greeks",
    "implied_vol",
    "black76",
    "black76_greeks",
    "american_option_lsm",
    "option_payoff",
]


def _validate_option_type(option_type: str) -> str:
    """Normalize and validate option_type parameter."""
    option_type = option_type.lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got '{option_type}'"
        )
    return option_type


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    r"""
    Compute the d1 term used in Black-Scholes and related models.

    .. math::

        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).

    Returns
    -------
    float
        The d1 value.
    """
    return (_np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * _np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    r"""
    Compute the d2 term used in Black-Scholes and related models.

    .. math::

        d_2 = d_1 - \sigma\sqrt{T}

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).

    Returns
    -------
    float
        The d2 value.
    """
    return _d1(S, K, T, r, sigma) - sigma * _np.sqrt(T)


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    r"""
    Price a European option using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.

    Returns
    -------
    float
        The Black-Scholes option price.

    Notes
    -----
    The Black-Scholes formula for a European call option is:

    .. math::

        C = S\,N(d_1) - K\,e^{-rT}\,N(d_2)

    and for a European put option:

    .. math::

        P = K\,e^{-rT}\,N(-d_2) - S\,N(-d_1)

    where

    .. math::

        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \quad
        d_2 = d_1 - \sigma\sqrt{T}

    and :math:`N(\cdot)` is the standard normal cumulative distribution function.

    Examples
    --------
    >>> import risktools as rt
    >>> rt.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    10.450583...
    >>> rt.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    5.573526...
    """
    option_type = _validate_option_type(option_type)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * _norm.cdf(d1) - K * _np.exp(-r * T) * _norm.cdf(d2)
    else:
        return K * _np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    r"""
    Compute all first-order Greeks for a European option under the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.

    Returns
    -------
    dict
        Dictionary with keys ``"delta"``, ``"gamma"``, ``"vega"``, ``"theta"``,
        and ``"rho"``.

    Notes
    -----
    The Greeks are defined as follows. Let :math:`\phi(\cdot)` denote the
    standard normal probability density function and :math:`N(\cdot)` the
    standard normal cumulative distribution function.

    **Delta** -- sensitivity of option price to the underlying price:

    .. math::

        \Delta_{\text{call}} = N(d_1), \quad
        \Delta_{\text{put}}  = N(d_1) - 1

    **Gamma** -- rate of change of delta with respect to the underlying price:

    .. math::

        \Gamma = \frac{\phi(d_1)}{S\,\sigma\sqrt{T}}

    **Vega** -- sensitivity to volatility (per unit change in :math:`\sigma`):

    .. math::

        \mathcal{V} = S\,\phi(d_1)\sqrt{T}

    To obtain vega per 1 percentage-point change in volatility, divide by 100.

    **Theta** -- sensitivity to the passage of time (per year):

    .. math::

        \Theta_{\text{call}} = -\frac{S\,\phi(d_1)\,\sigma}{2\sqrt{T}}
            - r\,K\,e^{-rT}\,N(d_2)

    .. math::

        \Theta_{\text{put}} = -\frac{S\,\phi(d_1)\,\sigma}{2\sqrt{T}}
            + r\,K\,e^{-rT}\,N(-d_2)

    **Rho** -- sensitivity to the risk-free rate:

    .. math::

        \rho_{\text{call}} = K\,T\,e^{-rT}\,N(d_2), \quad
        \rho_{\text{put}}  = -K\,T\,e^{-rT}\,N(-d_2)

    Examples
    --------
    >>> import risktools as rt
    >>> greeks = rt.black_scholes_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
    >>> greeks["delta"]
    0.6368...
    """
    option_type = _validate_option_type(option_type)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = _np.sqrt(T)
    pdf_d1 = _norm.pdf(d1)
    discount = _np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = _norm.cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * discount * _norm.cdf(d2)
        rho = K * T * discount * _norm.cdf(d2)
    else:
        delta = _norm.cdf(d1) - 1
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * discount * _norm.cdf(-d2)
        rho = -K * T * discount * _norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    r"""
    Compute the implied volatility of a European option via Newton-Raphson iteration.

    Given an observed market price, this function finds the volatility
    :math:`\sigma^*` such that the Black-Scholes price equals the observed price.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.
    tol : float, optional
        Convergence tolerance. Iteration stops when the absolute difference
        between the model price and observed price is less than ``tol``.
        By default ``1e-8``.
    max_iter : int, optional
        Maximum number of Newton-Raphson iterations. By default ``100``.

    Returns
    -------
    float
        The implied volatility :math:`\sigma^*`.

    Raises
    ------
    ValueError
        If the algorithm does not converge within ``max_iter`` iterations.

    Notes
    -----
    The Newton-Raphson update step is:

    .. math::

        \sigma_{n+1} = \sigma_n
            - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{mkt}}}
                   {\mathcal{V}(\sigma_n)}

    where :math:`C_{\text{BS}}(\sigma_n)` is the Black-Scholes price evaluated
    at volatility :math:`\sigma_n` and :math:`\mathcal{V}(\sigma_n)` is the
    Black-Scholes vega. The initial guess is :math:`\sigma_0 = 0.2`.

    Examples
    --------
    >>> import risktools as rt
    >>> bs_price = rt.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.25)
    >>> rt.implied_vol(price=bs_price, S=100, K=100, T=1, r=0.05)
    0.25000...
    """
    option_type = _validate_option_type(option_type)

    sigma = 0.2

    for i in range(max_iter):
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        diff = bs_price - price

        if abs(diff) < tol:
            return sigma

        vega = black_scholes_greeks(S, K, T, r, sigma, option_type)["vega"]

        if abs(vega) < 1e-15:
            # Vega is effectively zero; nudge sigma to avoid division by zero
            sigma += 0.01
            continue

        sigma = sigma - diff / vega

    raise ValueError(
        f"Implied volatility did not converge after {max_iter} iterations. "
        f"Last sigma={sigma:.6f}, price diff={diff:.6e}"
    )


def black76(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    r"""
    Price a European option on a futures or forward contract using the Black-76 model.

    Parameters
    ----------
    F : float
        Current futures or forward price.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the futures price (annualized).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.

    Returns
    -------
    float
        The Black-76 option price.

    Notes
    -----
    The Black-76 model adapts Black-Scholes for futures/forwards. The pricing
    formulas are:

    .. math::

        C = e^{-rT}\bigl[F\,N(d_1) - K\,N(d_2)\bigr]

    .. math::

        P = e^{-rT}\bigl[K\,N(-d_2) - F\,N(-d_1)\bigr]

    where

    .. math::

        d_1 = \frac{\ln(F/K) + \sigma^2 T / 2}{\sigma\sqrt{T}}, \quad
        d_2 = d_1 - \sigma\sqrt{T}

    Examples
    --------
    >>> import risktools as rt
    >>> rt.black76(F=100, K=100, T=1, r=0.05, sigma=0.2)
    7.5653...
    """
    option_type = _validate_option_type(option_type)

    d1 = (_np.log(F / K) + (sigma**2 / 2) * T) / (sigma * _np.sqrt(T))
    d2 = d1 - sigma * _np.sqrt(T)
    discount = _np.exp(-r * T)

    if option_type == "call":
        return discount * (F * _norm.cdf(d1) - K * _norm.cdf(d2))
    else:
        return discount * (K * _norm.cdf(-d2) - F * _norm.cdf(-d1))


def black76_greeks(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    r"""
    Compute all first-order Greeks for a European option on a futures contract
    under the Black-76 model.

    Parameters
    ----------
    F : float
        Current futures or forward price.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the futures price (annualized).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.

    Returns
    -------
    dict
        Dictionary with keys ``"delta"``, ``"gamma"``, ``"vega"``, ``"theta"``,
        and ``"rho"``.

    Notes
    -----
    Let :math:`D = e^{-rT}` denote the discount factor. The Greeks under
    Black-76 are:

    **Delta**:

    .. math::

        \Delta_{\text{call}} = D\,N(d_1), \quad
        \Delta_{\text{put}}  = -D\,N(-d_1)

    **Gamma**:

    .. math::

        \Gamma = \frac{D\,\phi(d_1)}{F\,\sigma\sqrt{T}}

    **Vega** (per unit change in :math:`\sigma`):

    .. math::

        \mathcal{V} = F\,D\,\phi(d_1)\sqrt{T}

    **Theta**:

    .. math::

        \Theta_{\text{call}} = -\frac{F\,D\,\phi(d_1)\,\sigma}{2\sqrt{T}}
            + r\,D\,\bigl[F\,N(d_1) - K\,N(d_2)\bigr] \cdot (-1)

    More precisely, since the entire payoff is discounted:

    .. math::

        \Theta_{\text{call}} = -\frac{F\,D\,\phi(d_1)\,\sigma}{2\sqrt{T}}
            - r\,C

    .. math::

        \Theta_{\text{put}} = -\frac{F\,D\,\phi(d_1)\,\sigma}{2\sqrt{T}}
            - r\,P

    where :math:`C` and :math:`P` are the call and put prices, respectively.

    **Rho**:

    .. math::

        \rho_{\text{call}} = -T\,C, \quad
        \rho_{\text{put}}  = -T\,P

    Examples
    --------
    >>> import risktools as rt
    >>> greeks = rt.black76_greeks(F=100, K=100, T=1, r=0.05, sigma=0.2)
    >>> greeks["delta"]
    0.5318...
    """
    option_type = _validate_option_type(option_type)

    d1 = (_np.log(F / K) + (sigma**2 / 2) * T) / (sigma * _np.sqrt(T))
    d2 = d1 - sigma * _np.sqrt(T)
    sqrt_T = _np.sqrt(T)
    discount = _np.exp(-r * T)
    pdf_d1 = _norm.pdf(d1)

    gamma = discount * pdf_d1 / (F * sigma * sqrt_T)
    vega = F * discount * pdf_d1 * sqrt_T

    price = black76(F, K, T, r, sigma, option_type)

    if option_type == "call":
        delta = discount * _norm.cdf(d1)
        theta = -(F * discount * pdf_d1 * sigma) / (2 * sqrt_T) - r * price
        rho = -T * price
    else:
        delta = -discount * _norm.cdf(-d1)
        theta = -(F * discount * pdf_d1 * sigma) / (2 * sqrt_T) - r * price
        rho = -T * price

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def american_option_lsm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    n_steps: int = 100,
    n_sims: int = 50000,
    seed: Optional[int] = None,
) -> float:
    r"""
    Price an American option using the Longstaff-Schwartz Least-Squares
    Monte Carlo (LSM) algorithm.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"put"``.
    n_steps : int, optional
        Number of time steps in the simulation. By default ``100``.
    n_sims : int, optional
        Number of simulation paths. By default ``50000``.
    seed : int, optional
        Random seed for reproducibility. By default ``None``.

    Returns
    -------
    float
        The estimated American option price.

    Notes
    -----
    The Longstaff-Schwartz algorithm proceeds as follows:

    1. Simulate :math:`M` paths of the underlying asset price under
       risk-neutral GBM dynamics:

       .. math::

           S_{t+\Delta t} = S_t \exp\Bigl[\bigl(r - \tfrac{\sigma^2}{2}\bigr)
           \Delta t + \sigma\sqrt{\Delta t}\,Z\Bigr]

       where :math:`Z \sim N(0,1)`.

    2. At expiration :math:`T`, compute the exercise payoff for each path.

    3. Working backwards from :math:`T - \Delta t` to :math:`\Delta t`:

       a. Identify in-the-money paths at step :math:`t`.

       b. Regress the discounted continuation values onto a polynomial basis
          of the current stock price (degree 2):

          .. math::

              E[V_{t+1} \mid S_t] \approx \beta_0 + \beta_1 S_t + \beta_2 S_t^2

       c. Exercise early at step :math:`t` if the immediate payoff exceeds
          the estimated continuation value.

    4. For each path, discount the (earliest) exercise payoff to time zero
       and return the average.

    Examples
    --------
    >>> import risktools as rt
    >>> price = rt.american_option_lsm(S=100, K=100, T=1, r=0.05, sigma=0.2,
    ...     option_type="put", n_steps=50, n_sims=10000, seed=42)
    """
    option_type = _validate_option_type(option_type)

    rng = _np.random.default_rng(seed)
    dt = T / n_steps
    discount_factor = _np.exp(-r * dt)

    # Step 1: Simulate GBM paths (n_steps+1 x n_sims)
    Z = rng.standard_normal((n_steps, n_sims))
    paths = _np.zeros((n_steps + 1, n_sims))
    paths[0] = S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * _np.exp(
            (r - sigma**2 / 2) * dt + sigma * _np.sqrt(dt) * Z[t - 1]
        )

    # Step 2: Compute payoff at expiration
    if option_type == "call":
        payoff = _np.maximum(paths[-1] - K, 0.0)
    else:
        payoff = _np.maximum(K - paths[-1], 0.0)

    # Cash flow matrix: the payoff received at the optimal exercise time
    # for each path, initialized to the terminal payoff
    cashflow = payoff.copy()
    exercise_time = _np.full(n_sims, n_steps, dtype=int)

    # Step 3: Backward induction
    for t in range(n_steps - 1, 0, -1):
        # Immediate exercise payoff at step t
        if option_type == "call":
            intrinsic = _np.maximum(paths[t] - K, 0.0)
        else:
            intrinsic = _np.maximum(K - paths[t], 0.0)

        # Only consider in-the-money paths for regression
        itm = intrinsic > 0
        if _np.sum(itm) == 0:
            continue

        # Discounted continuation values for ITM paths
        # Discount from the current exercise_time back to step t
        steps_to_exercise = exercise_time[itm] - t
        continuation = cashflow[itm] * _np.exp(-r * dt * steps_to_exercise)

        # Regression: continuation value ~ polynomial of stock price (degree 2)
        x = paths[t, itm]
        X = _np.column_stack([_np.ones_like(x), x, x**2])

        # Least-squares regression
        coeffs, _, _, _ = _np.linalg.lstsq(X, continuation, rcond=None)
        expected_continuation = X @ coeffs

        # Exercise decision: exercise if intrinsic > expected continuation
        exercise = intrinsic[itm] > expected_continuation
        exercise_indices = _np.where(itm)[0][exercise]

        # Update cash flows and exercise times for paths that exercise early
        cashflow[exercise_indices] = intrinsic[itm][exercise]
        exercise_time[exercise_indices] = t

    # Step 4: Discount all cash flows to time 0
    discount_to_zero = _np.exp(-r * dt * exercise_time)
    price = _np.mean(cashflow * discount_to_zero)

    return float(price)


def option_payoff(
    S: Union[float, _np.ndarray],
    K: float,
    option_type: str = "call",
    premium: float = 0.0,
) -> Union[float, _np.ndarray]:
    r"""
    Compute the payoff of a vanilla option position at expiration, net of
    the premium paid.

    Parameters
    ----------
    S : float or numpy.ndarray
        Price(s) of the underlying asset at expiration.
    K : float
        Strike price of the option.
    option_type : str, optional
        Type of option, either ``"call"`` or ``"put"``. By default ``"call"``.
    premium : float, optional
        Premium paid for the option. By default ``0.0``.

    Returns
    -------
    float or numpy.ndarray
        Net payoff(s) at expiration. Same shape as ``S``.

    Notes
    -----
    The payoff of a long call option at expiration is:

    .. math::

        \text{Payoff}_{\text{call}} = \max(S_T - K, 0) - \text{premium}

    The payoff of a long put option at expiration is:

    .. math::

        \text{Payoff}_{\text{put}} = \max(K - S_T, 0) - \text{premium}

    Examples
    --------
    >>> import numpy as np
    >>> import risktools as rt
    >>> S = np.linspace(80, 120, 5)
    >>> rt.option_payoff(S, K=100, option_type="call", premium=5.0)
    array([-5., -5., -5.,  5., 15.])
    """
    option_type = _validate_option_type(option_type)

    S = _np.asarray(S)

    if option_type == "call":
        payoff = _np.maximum(S - K, 0.0)
    else:
        payoff = _np.maximum(K - S, 0.0)

    result = payoff - premium

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result
