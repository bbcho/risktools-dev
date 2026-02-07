Options Pricing: The Binomial Tree Model
=========================================

This guide introduces the theory of option pricing through the lens of the
**binomial tree model**, one of the most intuitive and widely taught approaches
in quantitative finance. We begin with the definition of options and their
payoffs, develop the Cox-Ross-Rubinstein (CRR) framework step by step, show
how to use the ``crr_euro()`` function in **risktools**, and conclude with two
important theoretical results: convergence to the Black-Scholes formula and
put-call parity.

.. contents:: Topics covered
   :local:
   :depth: 2


1. Introduction to Options
--------------------------

An **option** is a financial derivative that gives its holder the *right*, but
not the *obligation*, to buy or sell an underlying asset at a predetermined
price on or before a specified date.

Calls and Puts
~~~~~~~~~~~~~~

There are two basic types of options:

* A **call option** gives the holder the right to *buy* the underlying asset at
  the strike price :math:`K`.
* A **put option** gives the holder the right to *sell* the underlying asset at
  the strike price :math:`K`.

At expiration, when the underlying asset price is :math:`S_T`, the payoffs are:

.. math::

   \text{Call payoff} &= \max(S_T - K,\; 0) \\[6pt]
   \text{Put payoff}  &= \max(K - S_T,\; 0)

A call is profitable when the asset price exceeds the strike; a put is
profitable when the asset price falls below it. In both cases the payoff is
never negative -- the holder simply lets the option expire worthless if
exercising would result in a loss.

Intrinsic Value vs. Time Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The price of an option can be decomposed into two parts:

* **Intrinsic value** -- the payoff the option would have if exercised
  immediately. For a call this is :math:`\max(S - K,\; 0)`; for a put it is
  :math:`\max(K - S,\; 0)`.
* **Time value** -- the additional premium above intrinsic value that reflects
  the possibility that the option may become more valuable before expiration.
  Time value is always non-negative and decays as the option approaches
  maturity (a phenomenon known as *theta decay*).

European vs. American Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A **European option** can only be exercised at maturity.
* An **American option** can be exercised at any time up to and including
  maturity.

Because an American option offers all the exercise opportunities of a European
option plus the additional right of early exercise, an American option is
always worth at least as much as an otherwise identical European option. For
non-dividend-paying stocks, it can be shown that early exercise of an American
*call* is never optimal, so the American and European call prices coincide
(Hull, 2018, Ch. 11).

The ``crr_euro()`` function in **risktools** prices **European** options.


2. The Binomial Model
---------------------

The binomial model, introduced by Cox, Ross, and Rubinstein (1979), provides a
discrete-time framework for option pricing that is both rigorous and
remarkably intuitive.

Core Idea
~~~~~~~~~

Divide the life of the option into :math:`n` equal time steps of length

.. math::

   \Delta t = \frac{T}{n}

where :math:`T` is the time to maturity in years. At each step the stock price
can move in one of two directions:

* **Up** by a multiplicative factor :math:`u`
* **Down** by a multiplicative factor :math:`d`

In the CRR parameterisation these factors are chosen to match the volatility
:math:`\sigma` of the underlying asset:

.. math::

   u &= e^{\,\sigma \sqrt{\Delta t}} \\[6pt]
   d &= e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}

The symmetry :math:`d = 1/u` ensures that an up-move followed by a down-move
returns the price to its original level, producing a *recombining* tree. This
is computationally important: after :math:`n` steps the tree has only
:math:`n + 1` distinct terminal nodes rather than :math:`2^n`.

Risk-Neutral Pricing
~~~~~~~~~~~~~~~~~~~~

Under the **risk-neutral measure**, all assets are expected to grow at the
risk-free rate :math:`r`. The risk-neutral probability of an up-move is:

.. math::

   q = \frac{e^{\,r\,\Delta t} - d}{u - d}

This is not a real-world probability; it is the probability that makes the
expected discounted payoff equal to the current price in a no-arbitrage
framework. The value :math:`q` must satisfy :math:`0 < q < 1` for the model to
exclude arbitrage opportunities.

Building the Tree
~~~~~~~~~~~~~~~~~

The pricing algorithm has two phases:

**Phase 1 -- Forward pass (asset prices).** Starting from the initial stock
price :math:`S_0`, build the tree forward. At time step :math:`i` and state
:math:`j` (where :math:`j` counts the number of up-moves), the asset price is:

.. math::

   S_{i,j} = S_0 \, u^{\,j} \, d^{\,i - j}, \qquad j = 0, 1, \ldots, i

**Phase 2 -- Backward pass (option values).** Starting from the terminal
payoffs at step :math:`n`, work backward through the tree. At maturity:

.. math::

   V_{n,j} = \max(S_{n,j} - K,\; 0) \qquad \text{(for a call)}

At each earlier node:

.. math::

   V_{i,j} = e^{-r\,\Delta t}\;\bigl[\,q\;V_{i+1,\,j+1} + (1-q)\;V_{i+1,\,j}\,\bigr]

The option price today is :math:`V_{0,0}`, the value at the root of the tree.

A Worked Example
~~~~~~~~~~~~~~~~

Consider a European call with :math:`S_0 = 100`, :math:`K = 100`,
:math:`\sigma = 0.20`, :math:`r = 0.10`, :math:`T = 1` year, and :math:`n = 2`
steps.

1. Compute :math:`\Delta t = 1/2 = 0.5`.
2. :math:`u = e^{0.20\sqrt{0.5}} \approx 1.1519`, :math:`d = 1/u \approx 0.8681`.
3. :math:`q = (e^{0.10 \times 0.5} - 0.8681) / (1.1519 - 0.8681) \approx 0.6442`.
4. The terminal asset prices at :math:`n = 2` are:

   - :math:`S_{2,2} = 100 \times u^2 \approx 132.69` (two up-moves)
   - :math:`S_{2,1} = 100 \times u \times d = 100.00` (one up, one down)
   - :math:`S_{2,0} = 100 \times d^2 \approx 75.36` (two down-moves)

5. Terminal call payoffs: :math:`\max(132.69 - 100, 0) = 32.69`,
   :math:`\max(100 - 100, 0) = 0`, :math:`\max(75.36 - 100, 0) = 0`.

6. Discount backward to find :math:`V_{0,0}`.


3. The ``crr_euro()`` Function
------------------------------

The **risktools** library provides a direct implementation of the CRR binomial
model through the :func:`crr_euro` function.

Parameters
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Default
     - Description
   * - ``s``
     - 100
     - Current stock (underlying asset) price.
   * - ``x``
     - 100
     - Strike price of the option.
   * - ``sigma``
     - 0.20
     - Annualised implied volatility of the underlying.
   * - ``Rf``
     - 0.10
     - Annualised risk-free interest rate.
   * - ``T``
     - 1
     - Time to maturity in years.
   * - ``n``
     - 5
     - Number of time steps in the binomial tree. Internally
       :math:`\Delta t = T / n`.
   * - ``type``
     - ``"call"``
     - Option type: ``"call"`` or ``"put"``.

Return Value
~~~~~~~~~~~~

``crr_euro()`` returns a dictionary with four keys:

* ``"asset"`` -- a NumPy array of shape :math:`(n+1) \times (n+1)` containing
  the asset price tree. Row :math:`i` corresponds to time step :math:`i`; for
  each row, columns :math:`0` through :math:`i` are populated.
* ``"option"`` -- a NumPy array of the same shape containing the option value
  tree, computed by backward induction.
* ``"price"`` -- a float giving the option price at time zero, i.e.
  ``option[0, 0]``.
* ``"note"`` -- a string indicating model validity. Returns ``"ok"`` when the
  condition :math:`\sigma > R_f \sqrt{\Delta t}` is satisfied, which ensures
  that the risk-neutral probability :math:`q` lies strictly between 0 and 1.

Model Validity
~~~~~~~~~~~~~~

For the CRR model to be arbitrage-free, the risk-neutral probability must
satisfy :math:`0 < q < 1`. This requires:

.. math::

   d < e^{\,r\,\Delta t} < u

which simplifies to the condition:

.. math::

   \sigma > R_f \sqrt{\Delta t}

When this condition is violated the ``"note"`` field in the returned dictionary
will contain a warning. In practice, this condition is almost always satisfied
for reasonable parameter choices and a sufficient number of time steps.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import risktools as rt

   # Price a European call option
   result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")

   print(f"Call price: {result['price']:.4f}")
   print(f"Model validity: {result['note']}")

.. code-block:: python

   # Inspect the asset price tree
   print("Asset price tree:")
   print(result["asset"].round(2))

   # Inspect the option value tree
   print("\nOption value tree:")
   print(result["option"].round(2))

.. code-block:: python

   # Price a European put with the same parameters
   put_result = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="put")

   print(f"Put price: {put_result['price']:.4f}")


4. Convergence to Black-Scholes
-------------------------------

One of the most important properties of the CRR model is that as the number of
time steps :math:`n \to \infty`, the binomial option price converges to the
continuous-time **Black-Scholes** price (Cox, Ross & Rubinstein, 1979).

The Black-Scholes Formula
~~~~~~~~~~~~~~~~~~~~~~~~~

For a European call on a non-dividend-paying stock, the Black-Scholes price is:

.. math::

   C = S\,\Phi(d_1) - K\,e^{-rT}\,\Phi(d_2)

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution
function, and

.. math::

   d_1 &= \frac{\ln(S/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}} \\[6pt]
   d_2 &= d_1 - \sigma\sqrt{T}

The corresponding European put price is:

.. math::

   P = K\,e^{-rT}\,\Phi(-d_2) - S\,\Phi(-d_1)

Observing Convergence
~~~~~~~~~~~~~~~~~~~~~

We can observe convergence by computing the CRR price for increasing values of
:math:`n` and comparing it with the Black-Scholes closed-form solution.

.. code-block:: python

   import numpy as np
   from scipy.stats import norm
   import risktools as rt

   # Parameters
   S, K, sigma, r, T = 100, 100, 0.20, 0.10, 1.0

   # Black-Scholes analytical price
   d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
   d2 = d1 - sigma * np.sqrt(T)
   bs_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

   print(f"Black-Scholes call price: {bs_call:.4f}")

   # CRR prices for increasing n
   for n in [5, 10, 25, 50, 100, 200, 500]:
       crr = rt.crr_euro(s=S, x=K, sigma=sigma, Rf=r, T=T, n=n, type="call")
       error = crr["price"] - bs_call
       print(f"  n = {n:>4d}:  CRR price = {crr['price']:.4f},  error = {error:+.4f}")

As :math:`n` grows the error shrinks, confirming that the discrete binomial
lattice converges to the continuous diffusion limit. The convergence is of
order :math:`O(1/n)`, with a characteristic oscillation pattern: odd and even
values of :math:`n` approach the Black-Scholes price from opposite sides.

**Why does this happen?** The CRR up and down factors are constructed so that
the log-returns of the underlying follow a binomial distribution that
approximates a normal distribution (by the Central Limit Theorem) as
:math:`n \to \infty`. In the limit, the binomial random walk becomes a
geometric Brownian motion -- precisely the assumption underlying the
Black-Scholes model (Black & Scholes, 1973).


5. Put-Call Parity
------------------

**Put-call parity** is a fundamental no-arbitrage relationship linking the
prices of European calls and puts with the same strike and maturity. It was
first formalised by Stoll (1969).

The Relationship
~~~~~~~~~~~~~~~~

For European options on a non-dividend-paying stock:

.. math::

   C - P = S - K\,e^{-rT}

where :math:`C` and :math:`P` are the call and put prices, :math:`S` is the
current stock price, :math:`K` is the common strike price, :math:`r` is the
risk-free rate, and :math:`T` is the time to maturity.

**Intuition.** A portfolio consisting of a long call and a short put with the
same strike and maturity replicates a forward contract to buy the stock at
price :math:`K` at time :math:`T`. The present value of that obligation is
:math:`S - K\,e^{-rT}`.

Verifying with ``crr_euro()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can verify that the CRR model respects put-call parity:

.. code-block:: python

   import numpy as np
   import risktools as rt

   # Common parameters
   S, K, sigma, r, T, n = 100, 105, 0.25, 0.05, 1.0, 200

   # Price call and put
   call = rt.crr_euro(s=S, x=K, sigma=sigma, Rf=r, T=T, n=n, type="call")
   put  = rt.crr_euro(s=S, x=K, sigma=sigma, Rf=r, T=T, n=n, type="put")

   # Put-call parity: C - P should equal S - K*exp(-r*T)
   lhs = call["price"] - put["price"]
   rhs = S - K * np.exp(-r * T)

   print(f"Call price (C):       {call['price']:.4f}")
   print(f"Put price (P):        {put['price']:.4f}")
   print(f"C - P:                {lhs:.4f}")
   print(f"S - K*exp(-rT):       {rhs:.4f}")
   print(f"Difference:           {abs(lhs - rhs):.6f}")

The difference should be very small (on the order of :math:`10^{-10}` or less
for the binomial model, which satisfies put-call parity exactly at each node
by construction).

**Note.** Put-call parity holds exactly for European options. For American
options the relationship becomes an inequality:

.. math::

   S - K \;\leq\; C_{\text{Am}} - P_{\text{Am}} \;\leq\; S - K\,e^{-rT}


References
----------

Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. *Journal of Political Economy*, 81(3), 637--654.

Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). Option Pricing: A Simplified
Approach. *Journal of Financial Economics*, 7(3), 229--263.

Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.).
Pearson.

Stoll, H. R. (1969). The Relationship Between Put and Call Option Prices.
*Journal of Finance*, 24(5), 801--824.
