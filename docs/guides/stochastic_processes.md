# Stochastic Processes for Commodities

This guide introduces the stochastic processes used in commodity price modeling
and implemented in **risktools**. We begin with the theoretical foundations of
random walks in financial markets, then build progressively from the simplest
diffusion model (Geometric Brownian Motion) through mean-reverting processes
(Ornstein--Uhlenbeck) and finally to jump-diffusion models that capture the
sudden dislocations common in energy and commodity markets.

The presentation emphasizes both the mathematical structure of each process and
its practical implementation in `risktools`, so that students can move
fluently between theory and computation.


## Introduction to Stochastic Processes in Finance

### Why Random Walks?

A central question in quantitative finance is: *how should we model the future
evolution of asset prices?* The answer, perhaps surprisingly, begins with the
observation that prices in well-functioning markets are inherently
unpredictable.

The **Efficient Market Hypothesis** (EMH), articulated in its modern form by
Fama (1970), asserts that asset prices fully reflect all available information.
If this is the case, then price changes can only be driven by the arrival of
*new* information, which by definition is unpredictable. The mathematical
object that formalizes this unpredictability is a **stochastic process** -- a
collection of random variables indexed by time.

More precisely, if markets are efficient, then future price changes are
uncorrelated with past price changes, and the best forecast of tomorrow's price
is simply today's price plus a random shock. This is the essence of a **random
walk**:

$$
S_{t+1} = S_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \mathcal{N}(0, \sigma^2)
$$

While the EMH was originally developed for equity markets, commodity markets
exhibit important departures from this framework. Commodity prices are
influenced by storage costs, convenience yields, seasonal supply and demand
patterns, and physical constraints that create **mean-reverting** behavior.
Understanding when and why commodity prices depart from a pure random walk is
the central theme of this guide.

### From Discrete to Continuous Time

In practice, we observe prices at discrete intervals (daily, hourly, etc.), but
the mathematical theory is most elegantly expressed in continuous time using
**stochastic differential equations** (SDEs). An SDE has the general form:

$$
dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t
$$

where \(W_t\) is a **Wiener process** (standard Brownian motion) satisfying:

1. \(W_0 = 0\)
2. \(W_t - W_s \sim \mathcal{N}(0, t-s)\) for \(t > s\)
3. Increments over non-overlapping intervals are independent

The term \(a(X_t, t)\,dt\) is the **drift** (deterministic trend), and
\(b(X_t, t)\,dW_t\) is the **diffusion** (random fluctuation). Different
choices of the functions \(a\) and \(b\) yield different stochastic
processes, each suited to different modeling contexts.

To simulate these continuous-time processes on a computer, we discretize them
using numerical schemes such as the **Euler--Maruyama method**, which replaces
infinitesimal increments with small but finite time steps \(\Delta t\).


## Geometric Brownian Motion (GBM)

### The Model

**Geometric Brownian Motion** is the foundational model for asset price
dynamics in modern finance. It was used by Black and Scholes (1973) to derive
their celebrated option pricing formula and remains the standard starting
point for price modeling.

The SDE for GBM is:

$$
dS_t = \mu\, S_t\, dt + \sigma\, S_t\, dW_t
$$

where:

- \(S_t\) is the asset price at time \(t\),
- \(\mu\) is the drift rate (expected return per unit time),
- \(\sigma\) is the volatility (annualized standard deviation of returns),
- \(W_t\) is a standard Wiener process.

The key feature of this SDE is that both the drift and diffusion are
**proportional to the current price** \(S_t\). This ensures that prices
remain strictly positive and that percentage returns (rather than absolute
changes) are normally distributed.

### Analytical Solution

Applying Ito's lemma to \(\ln S_t\), we obtain the exact solution:

$$
S(t) = S(0) \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma\, W(t)\right]
$$

This tells us several important things:

- \(\ln S(t)\) is normally distributed, so \(S(t)\) is
  **log-normally distributed**.
- The expected value is \(\mathbb{E}[S(t)] = S(0)\,e^{\mu t}\).
- The term \(-\sigma^2/2\) is the **Ito correction**, arising from the
  fact that the exponential of a normal random variable has a higher mean than
  the exponential of the mean. Without this correction, the discretized
  simulation would exhibit systematic upward bias.

### Euler--Maruyama Discretization

To simulate GBM numerically, we use the **Euler--Maruyama** discretization.
Rather than discretizing the SDE for \(S_t\) directly (which would require
multiplicative noise), we discretize the log-price process. Over a small time
step \(\Delta t\), the log return is:

$$
\ln S_{t+\Delta t} - \ln S_t = \left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\,\sqrt{\Delta t}\;\varepsilon_t
$$

where \(\varepsilon_t \sim \mathcal{N}(0, 1)\). Exponentiating gives the
multiplicative update:

$$
S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\,\sqrt{\Delta t}\;\varepsilon_t\right]
$$

This is precisely the scheme implemented in `simGBM()`. By
working with log returns and using `cumprod`, the implementation avoids a
slow Python loop and computes all time steps in a vectorized fashion.

### Properties and Limitations

**Properties of GBM:**

- Prices are always positive.
- Log returns are independent and identically distributed (i.i.d.) normal.
- The process is a **martingale** under the risk-neutral measure (when
  \(\mu = r\), the risk-free rate).
- The variance of \(S(t)\) grows without bound as \(t \to \infty\).

**Limitations for commodity modeling:**

- **No mean reversion.** GBM assumes prices can drift arbitrarily far from any
  equilibrium level. For commodities, the cost of production and substitution
  effects create long-run price anchors.
- **No jumps.** GBM produces continuous sample paths, but commodity prices
  often exhibit sudden jumps due to supply disruptions, weather events, or
  policy changes.
- **Constant volatility.** GBM assumes volatility is constant, while commodity
  volatility often varies with price level and season.

Despite these limitations, GBM is a useful benchmark and appropriate for
short-horizon simulations where mean reversion has little time to act.

### Simulation with `simGBM`

The function `simGBM()` simulates GBM paths using the
Euler--Maruyama discretization described above.

**Function signature:**

```python
simGBM(s0=10, mu=0, sigma=0.2, r=0, T=1, dt=1/252, sims=1000, eps=None)
```

**Parameters:**

- `s0` -- initial price at time zero
- `mu` -- mean of the normally distributed random shocks
- `sigma` -- annualized volatility
- `r` -- risk-free interest rate (used in the drift)
- `T` -- time horizon in years
- `dt` -- time step as a fraction of a year (e.g., `1/252` for one
  business day)
- `sims` -- number of Monte Carlo paths
- `eps` -- optional matrix of pre-generated random shocks (useful for
  reproducibility and variance reduction)

**Example: Simulating 5 GBM paths over one year.**

```python
import risktools as rt
import matplotlib.pyplot as plt

# Simulate 5 paths of GBM
paths = rt.simGBM(s0=100, mu=0, sigma=0.3, r=0.05, T=1, dt=1/252, sims=5)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
paths.plot(ax=ax, legend=False, alpha=0.8)
ax.set_xlabel("Time Step (business days)")
ax.set_ylabel("Price ($)")
ax.set_title("Geometric Brownian Motion: 5 Sample Paths")
ax.axhline(y=100, color="black", linestyle="--", linewidth=0.8, label="$S_0$")
ax.legend()
plt.tight_layout()
plt.show()
```

**Example: Examining the distribution of terminal prices.**

A key prediction of GBM is that terminal prices are log-normally distributed.
We can verify this with a large simulation:

```python
import numpy as np

# Simulate 10,000 paths
paths = rt.simGBM(s0=100, mu=0, sigma=0.3, r=0.05, T=1, dt=1/252, sims=10000)
terminal_prices = paths.iloc[-1, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left panel: histogram of terminal prices (log-normal)
axes[0].hist(terminal_prices, bins=80, density=True, alpha=0.7, edgecolor="white")
axes[0].set_xlabel("Terminal Price")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution of $S(T)$ (Log-Normal)")

# Right panel: histogram of log returns (normal)
log_returns = np.log(terminal_prices / 100)
axes[1].hist(log_returns, bins=80, density=True, alpha=0.7, edgecolor="white")
axes[1].set_xlabel("Log Return")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution of $\\ln(S(T)/S(0))$ (Normal)")

plt.tight_layout()
plt.show()
```


## The Ornstein--Uhlenbeck (OU) Process

### Motivation: Mean Reversion in Commodity Markets

Many commodity prices and spreads exhibit **mean reversion**: when the price
deviates from a long-run equilibrium level, economic forces tend to push it
back. Examples include:

- **Crack spreads** (the difference between refined product and crude oil
  prices) revert because refinery margins attract or deter production.
- **Calendar spreads** (the difference between nearby and deferred futures)
  revert due to storage economics and the theory of normal backwardation.
- **Basis differentials** (the difference between prices at two locations)
  revert as transportation and arbitrage eliminate spatial dislocations.

For these quantities, GBM is inappropriate because it allows the process to
wander arbitrarily far from equilibrium. We need a model that incorporates a
**restoring force** -- the Ornstein--Uhlenbeck process.

### The SDE

The **Ornstein--Uhlenbeck process** is defined by the SDE:

$$
dX_t = \theta\,(\mu - X_t)\,dt + \sigma\,dW_t
$$

where:

- \(X_t\) is the process value (e.g., a spread level) at time \(t\),
- \(\mu\) is the **long-run mean** to which the process reverts,
- \(\theta > 0\) is the **mean-reversion speed** (higher values mean
  faster reversion),
- \(\sigma\) is the volatility,
- \(W_t\) is a standard Wiener process.

The drift term \(\theta(\mu - X_t)\) acts as a spring:

- When \(X_t > \mu\), the drift is negative, pulling the process down.
- When \(X_t < \mu\), the drift is positive, pushing the process up.
- The strength of the pull is proportional to both \(\theta\) and the
  deviation \(|X_t - \mu|\).

!!! note

    Unlike GBM, the diffusion term in the OU process is **additive** (not
    proportional to \(X_t\)). This means the OU process can theoretically
    take negative values, which is acceptable for spreads and log-prices but
    not for raw prices. When modeling prices directly, one often applies the
    OU process to the logarithm of the price.

### Euler--Maruyama Discretization

Discretizing the OU SDE over a time step \(\Delta t\):

$$
X_{t+\Delta t} = X_t + \theta\,(\mu - X_t)\,\Delta t + \sigma\,\sqrt{\Delta t}\;\varepsilon_t
$$

where \(\varepsilon_t \sim \mathcal{N}(0, 1)\). This is the standard
Euler--Maruyama scheme, and it is the update rule used in the inner loop of
`simOU()`.

### Half-Life of Mean Reversion

An important practical quantity is the **half-life** of mean reversion -- the
expected time for the process to close half of the distance between its
current value and the long-run mean. From the deterministic part of the SDE
(\(\sigma = 0\)):

$$
X_t = \mu + (X_0 - \mu)\,e^{-\theta t}
$$

Setting \(X_t - \mu = \tfrac{1}{2}(X_0 - \mu)\) and solving for \(t\):

$$
t_{1/2} = \frac{\ln 2}{\theta}
$$

For example, if \(\theta = 2\) (per year), the half-life is
\(\ln 2 / 2 \approx 0.347\) years, or about 87 business days. The
half-life provides an intuitive way to interpret the speed of mean reversion
and to compare estimates across different datasets.

### Stationary Distribution

A remarkable property of the OU process is that it possesses a **stationary
distribution**. As \(t \to \infty\), the distribution of \(X_t\)
converges to:

$$
X_\infty \sim \mathcal{N}\!\left(\mu,\; \frac{\sigma^2}{2\theta}\right)
$$

This result has two important implications:

1. **The variance is finite** and determined by the ratio
   \(\sigma^2 / (2\theta)\). Faster mean reversion (larger \(\theta\))
   or lower volatility (smaller \(\sigma\)) produces a tighter
   distribution around \(\mu\).
2. **The process is ergodic**: time averages converge to ensemble averages,
   which justifies estimating \(\mu\) from a single long time series.

### Simulation with `simOU`

The function `simOU()` simulates OU paths with support for
time-varying and stochastic mean and volatility parameters.

**Function signature:**

```python
simOU(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1/252, sims=1000,
      eps=None, seed=None, log_price=False, c=True)
```

**Key parameters:**

- `s0` -- starting value at time zero
- `mu` -- long-run mean (scalar, 1D array for time-varying, or 2D array for
  stochastic mean)
- `theta` -- mean-reversion speed
- `sigma` -- annualized volatility (also supports 1D and 2D arrays)
- `T` -- time horizon in years
- `dt` -- time step as a fraction of a year
- `sims` -- number of Monte Carlo paths
- `seed` -- random seed for reproducibility
- `log_price` -- if `True`, adds an Ito correction term for log-price
  dynamics
- `c` -- if `True` (default), uses the C-optimized backend for speed

**Example: Simulating mean-reverting paths and visualizing reversion.**

```python
import risktools as rt
import matplotlib.pyplot as plt

# Simulate OU process starting far from the mean
paths = rt.simOU(s0=8, mu=5, theta=2, sigma=0.5, T=2, dt=1/252, sims=5, seed=42)

fig, ax = plt.subplots(figsize=(10, 5))
paths.plot(ax=ax, legend=False, alpha=0.8)
ax.axhline(y=5, color="red", linestyle="--", linewidth=1.5, label="$\\mu = 5$")
ax.set_xlabel("Time Step (business days)")
ax.set_ylabel("Spread Level")
ax.set_title("Ornstein-Uhlenbeck Process: Mean Reversion from $X_0 = 8$ to $\\mu = 5$")
ax.legend()
plt.tight_layout()
plt.show()
```

**Example: Verifying the stationary distribution.**

For a large number of long-horizon simulations, the distribution of terminal
values should match the theoretical stationary distribution:

```python
import numpy as np
from scipy import stats

# Parameters
mu, theta, sigma = 5, 2, 0.5

# Simulate many paths for a long horizon
paths = rt.simOU(s0=5, mu=mu, theta=theta, sigma=sigma,
                  T=5, dt=1/252, sims=10000, seed=123)

terminal_values = paths.iloc[-1, :]

# Theoretical stationary distribution
stationary_std = sigma / np.sqrt(2 * theta)
x_grid = np.linspace(mu - 4*stationary_std, mu + 4*stationary_std, 200)
theoretical_pdf = stats.norm.pdf(x_grid, loc=mu, scale=stationary_std)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(terminal_values, bins=80, density=True, alpha=0.6,
        edgecolor="white", label="Simulated")
ax.plot(x_grid, theoretical_pdf, "r-", linewidth=2,
        label=f"$N(\\mu={mu},\\; \\sigma^2/2\\theta={stationary_std**2:.3f})$")
ax.set_xlabel("Terminal Value")
ax.set_ylabel("Density")
ax.set_title("OU Stationary Distribution: Theory vs. Simulation")
ax.legend()
plt.tight_layout()
plt.show()
```


## OU with Jump Diffusion (OUJ)

### Motivation: Sudden Dislocations in Commodity Markets

While the OU process captures the day-to-day mean-reverting behavior of
commodity spreads, it cannot generate the **sudden, large dislocations** that
are a hallmark of energy markets. These jumps arise from:

- **Pipeline outages** -- unexpected shutdowns that create immediate supply
  shortfalls at delivery points.
- **Refinery turnarounds** -- planned or unplanned maintenance that removes
  processing capacity and widens crack spreads.
- **Weather events** -- polar vortices, hurricanes, or heat waves that cause
  demand spikes or supply disruptions.
- **Geopolitical shocks** -- sanctions, export bans, or conflicts that
  suddenly restrict supply.

After such an event, the spread does not instantly revert to its long-run
mean. Instead, the market adjusts gradually as alternative supply is arranged,
inventories are drawn, or demand substitution occurs. This motivates the
addition of a **Poisson jump component** to the OU process, along with an
optional **mean-reversion lag** to capture the persistence of jump effects.

### The SDE

The **Ornstein--Uhlenbeck Jump Diffusion** (OUJ) process combines mean
reversion with a compound Poisson jump:

$$
dX_t = \theta\,(\mu - \lambda\,\bar{J} - X_t)\,X_t\,dt + \sigma\,X_t\,dW_t + J_t\,dN_t
$$

where:

- \(\theta\), \(\mu\), \(\sigma\), and \(W_t\) have the same
  interpretation as in the standard OU process,
- \(N_t\) is a **Poisson process** with intensity \(\lambda\)
  (the probability of a jump per unit time),
- \(J_t\) is the **jump size**, drawn from a log-normal distribution with
  mean \(\bar{J}\) and standard deviation \(\sigma_J\),
- the drift is adjusted by \(-\lambda\,\bar{J}\) to compensate for the
  expected jump contribution, keeping the long-run mean centered at
  \(\mu\).

The Poisson process \(N_t\) generates random arrival times for jumps.
In any infinitesimal interval \([t, t+dt)\):

$$
\Pr(dN_t = 1) = \lambda\,dt, \qquad \Pr(dN_t = 0) = 1 - \lambda\,dt
$$

When a jump occurs (\(dN_t = 1\)), the process experiences an
instantaneous shift of size \(J_t\). Between jumps, the process follows
the standard OU dynamics.

This framework was originally proposed by Merton (1976) for equity options
pricing and has since been widely adopted in commodity modeling where jump
risk is a first-order concern.

### Mean-Reversion Lag

A distinctive feature of the `risktools` implementation is the optional
**mean-reversion lag** parameter (`mr_lag`). In many commodity markets, the
price does not begin reverting immediately after a jump. For example:

- A pipeline outage may take several weeks to repair, during which the
  spread remains elevated.
- A refinery turnaround has a known duration, and the market prices in the
  full period of reduced capacity.

The `mr_lag` parameter specifies the number of time steps during which the
mean-reversion level is shifted by the jump amount. Concretely, when a jump
of size \(J\) occurs at time \(t\):

$$
\mu_{\text{eff}}(s) = \mu + J \quad \text{for } s \in [t, t + \texttt{mr\_lag} \cdot \Delta t)
$$

After the lag period, the effective mean reverts to \(\mu\), and the
process begins pulling back toward the original equilibrium. This mechanism
produces realistic **plateau-then-revert** dynamics following a jump event.

### Simulation with `simOUJ`

The function `simOUJ()` simulates OU jump-diffusion paths.

**Function signature:**

```python
simOUJ(s0=5, mu=5, theta=0.5, sigma=0.2, jump_prob=0.05,
       jump_avgsize=3, jump_stdv=0.05, T=1, dt=1/12, sims=1000,
       mr_lag=None, eps=None, elp=None, ejp=None, seed=None, c=True)
```

**Key parameters:**

- `s0` -- initial value
- `mu` -- long-run mean (scalar or array for time-varying mean)
- `theta` -- mean-reversion speed
- `sigma` -- annualized volatility (scalar, 1D, or 2D array)
- `jump_prob` -- jump intensity \(\lambda\) (probability per unit time)
- `jump_avgsize` -- mean jump size \(\bar{J}\) (log-normal mean)
- `jump_stdv` -- standard deviation of jump size (log-normal \(\sigma_J\))
- `T` -- time horizon in years
- `dt` -- time step as a fraction of a year
- `sims` -- number of Monte Carlo paths
- `mr_lag` -- number of time steps that mean reversion is delayed after a
  jump (`None` for no delay)
- `seed` -- random seed for reproducibility

**Example: Simulating jump-diffusion paths with monthly steps.**

```python
import risktools as rt
import matplotlib.pyplot as plt

# OU with jumps: monthly time steps over 3 years
paths = rt.simOUJ(
    s0=5, mu=5, theta=0.5, sigma=0.2,
    jump_prob=0.05, jump_avgsize=3, jump_stdv=0.05,
    T=3, dt=1/12, sims=5, seed=42
)

fig, ax = plt.subplots(figsize=(10, 5))
paths.plot(ax=ax, legend=False, alpha=0.8)
ax.axhline(y=5, color="red", linestyle="--", linewidth=1.5, label="$\\mu = 5$")
ax.set_xlabel("Time Step (months)")
ax.set_ylabel("Spread Level")
ax.set_title("OU Jump Diffusion: Jumps with Immediate Mean Reversion")
ax.legend()
plt.tight_layout()
plt.show()
```

**Example: Comparing paths with and without mean-reversion lag.**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Without lag
paths_no_lag = rt.simOUJ(
    s0=5, mu=5, theta=0.5, sigma=0.2,
    jump_prob=0.1, jump_avgsize=2, jump_stdv=0.05,
    T=3, dt=1/12, sims=3, mr_lag=None, seed=99
)
paths_no_lag.plot(ax=axes[0], legend=False, alpha=0.8)
axes[0].axhline(y=5, color="red", linestyle="--", linewidth=1.5)
axes[0].set_title("No Mean-Reversion Lag (mr_lag=None)")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("Spread Level")

# With lag of 4 months
paths_with_lag = rt.simOUJ(
    s0=5, mu=5, theta=0.5, sigma=0.2,
    jump_prob=0.1, jump_avgsize=2, jump_stdv=0.05,
    T=3, dt=1/12, sims=3, mr_lag=4, seed=99
)
paths_with_lag.plot(ax=axes[1], legend=False, alpha=0.8)
axes[1].axhline(y=5, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("With Mean-Reversion Lag (mr_lag=4)")
axes[1].set_xlabel("Time Step")

plt.suptitle("Effect of Mean-Reversion Lag on Jump-Diffusion Paths", fontsize=13)
plt.tight_layout()
plt.show()
```

When `mr_lag` is active, the simulated paths will show a characteristic
**plateau** after a jump before gradually reverting to the long-run mean. This
is especially useful for modeling refinery turnarounds with a known duration or
pipeline outages with an estimated repair timeline.


## Parameter Estimation

### Theoretical Background

Given an observed time series that we believe follows an OU process, we need
to estimate the parameters \(\theta\), \(\mu\), and \(\sigma\).
The function `fitOU()` provides two estimation methods: **OLS
regression** and **maximum likelihood estimation** (MLE).

### OLS Method

The OLS method exploits the discrete-time representation of the OU process.
Consider the Euler--Maruyama discretization:

$$
X_{t+\Delta t} - X_t = \theta\,(\mu - X_t)\,\Delta t + \sigma\,\sqrt{\Delta t}\;\varepsilon_t
$$

Rearranging:

$$
\Delta X_t = a + b\,X_t + \text{noise}
$$

where \(\Delta X_t = X_{t+\Delta t} - X_t\). This is a linear regression
of the changes \(\Delta X_t\) on the levels \(X_t\), with:

$$
a = \theta\,\mu\,\Delta t, \qquad b = -\theta\,\Delta t
$$

From the OLS estimates \(\hat{a}\) and \(\hat{b}\):

$$
\hat{\theta} = -\frac{\hat{b}}{\Delta t}, \qquad
\hat{\mu} = \frac{\hat{a}}{\hat{\theta}\,\Delta t}, \qquad
\hat{\sigma} = \frac{\text{std}(\hat{\varepsilon})}{\sqrt{\Delta t}}
$$

where \(\hat{\varepsilon}\) denotes the OLS residuals. The OLS method is
simple, transparent, and provides standard errors and diagnostic statistics
through the underlying regression output (accessible via `verbose=True`).

### MLE Method

The MLE method directly maximizes the log-likelihood of the observed data
under the exact transition density of the OU process. The conditional
distribution of \(X_{t+\delta}\) given \(X_t\) is:

$$
X_{t+\delta} \mid X_t \sim \mathcal{N}\!\left(
    \mu + (X_t - \mu)\,e^{-\theta\delta},\;
    \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\delta}\right)
\right)
$$

The MLE uses sufficient statistics \(S_x\), \(S_y\), \(S_{xx}\),
\(S_{yy}\), and \(S_{xy}\) computed from the observations to obtain
closed-form expressions for the parameter estimates. The MLE is
asymptotically efficient (it achieves the Cramer--Rao lower bound) and does
not depend on the choice of \(\Delta t\) in the same way as OLS.

### Estimation with `fitOU`

**Function signature:**

```python
fitOU(spread, dt=1/252, log_price=False, method="OLS", verbose=False)
```

**Parameters:**

- `spread` -- a 1D array-like (list, Series, or single DataFrame column) of
  the observed OU process
- `dt` -- time step as a fraction of a year (only used for OLS)
- `log_price` -- if `True`, takes the natural logarithm of the input
  before estimation
- `method` -- `"OLS"` or `"MLE"`
- `verbose` -- if `True` and method is `"OLS"`, prints the full
  regression summary

**Returns:** a dictionary with keys `"theta"`, `"mu"`, and
`"annualized_sigma"`.

**Example: Simulate an OU process and recover the parameters.**

```python
import risktools as rt
import numpy as np

# True parameters
true_mu = 5.0
true_theta = 2.0
true_sigma = 0.5

# Simulate one long path
paths = rt.simOU(s0=5, mu=true_mu, theta=true_theta, sigma=true_sigma,
                  T=10, dt=1/252, sims=1, seed=42)

# Extract the single path as a Series
spread = paths.iloc[:, 0]

# Estimate with OLS
params_ols = rt.fitOU(spread, dt=1/252, method="OLS")
print("OLS estimates:")
print(f"  theta = {params_ols['theta']:.4f}  (true: {true_theta})")
print(f"  mu    = {params_ols['mu']:.4f}  (true: {true_mu})")
print(f"  sigma = {params_ols['annualized_sigma']:.4f}  (true: {true_sigma})")

# Estimate with MLE
params_mle = rt.fitOU(spread, method="MLE")
print("\nMLE estimates:")
print(f"  theta = {params_mle['theta']:.4f}  (true: {true_theta})")
print(f"  mu    = {params_mle['mu']:.4f}  (true: {true_mu})")
print(f"  sigma = {params_mle['annualized_sigma']:.4f}  (true: {true_sigma})")
```

**Example: Verifying parameter recovery across many simulations.**

To assess estimator quality, we can simulate many independent OU paths and
examine the distribution of parameter estimates:

```python
import pandas as pd

results = []
for i in range(200):
    path = rt.simOU(s0=5, mu=5.0, theta=2.0, sigma=0.5,
                     T=5, dt=1/252, sims=1, seed=i)
    est = rt.fitOU(path.iloc[:, 0], dt=1/252, method="MLE")
    results.append(est)

df = pd.DataFrame(results)
print("Parameter recovery (200 simulations, T=5 years):")
print(df.describe().round(4))
```

### Practical Considerations

When applying `fitOU` to real data, keep the following points in mind:

1. **Data frequency and** `dt`. The `dt` parameter must match the actual
   sampling frequency of the data. For daily business-day data, use
   `dt=1/252`. Misspecifying `dt` will scale \(\hat{\theta}\) and
   \(\hat{\sigma}\) incorrectly.

2. **Sample size.** Mean-reversion parameters are notoriously difficult to
   estimate from short samples. As a rule of thumb, the time series should
   span at least several half-lives (\(T \gg \ln 2 / \theta\)) to obtain
   reliable estimates. With daily data and \(\theta \approx 2\), this
   means at least 2--3 years of data.

3. **Stationarity.** The OU model assumes the process is stationary (or at
   least covariance-stationary). If the long-run mean \(\mu\) is shifting
   over time (due to structural changes in the market), the estimates will be
   biased. Consider using rolling-window estimation or allowing for a
   time-varying mean.

4. **OLS vs. MLE.** For large samples with small \(\Delta t\), both
   methods give similar results. The MLE is more efficient in small samples
   and does not depend on the Euler--Maruyama approximation. The OLS method
   is more transparent and provides regression diagnostics.

5. **Outliers and jumps.** If the data contains jumps (as in an OUJ process),
   the standard OU estimators will be biased -- jumps inflate the estimated
   volatility and may distort the mean-reversion speed. Pre-filtering jumps
   or using robust estimation techniques is advisable in such cases.


## Putting It All Together

The following example demonstrates a complete workflow: simulate an OU process,
estimate its parameters, and use the estimated parameters to generate new
forecasts.

```python
import risktools as rt
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Simulate an OU process (stand-in for observed market data)
true_params = {"mu": 4.0, "theta": 1.5, "sigma": 0.4}
observed = rt.simOU(
    s0=4.5, mu=true_params["mu"], theta=true_params["theta"],
    sigma=true_params["sigma"], T=5, dt=1/252, sims=1, seed=7
)
spread = observed.iloc[:, 0]

# Step 2: Estimate parameters
est = rt.fitOU(spread, dt=1/252, method="MLE")
print("Estimated parameters:")
for k, v in est.items():
    print(f"  {k}: {v:.4f}")

# Step 3: Compute half-life
half_life_days = np.log(2) / est["theta"] * 252
print(f"\nEstimated half-life: {half_life_days:.0f} business days")

# Step 4: Forward simulation from the last observed value
last_value = spread.iloc[-1]
forecasts = rt.simOU(
    s0=last_value, mu=est["mu"], theta=est["theta"],
    sigma=est["annualized_sigma"], T=1, dt=1/252, sims=500, seed=0
)

# Step 5: Plot historical data and fan chart of forecasts
fig, ax = plt.subplots(figsize=(12, 5))

# Historical
t_hist = np.arange(len(spread))
ax.plot(t_hist, spread.values, "k-", linewidth=1, label="Observed")

# Forecast fan chart
t_fwd = np.arange(len(spread) - 1, len(spread) - 1 + forecasts.shape[0])
q05 = forecasts.quantile(0.05, axis=1)
q25 = forecasts.quantile(0.25, axis=1)
q50 = forecasts.quantile(0.50, axis=1)
q75 = forecasts.quantile(0.75, axis=1)
q95 = forecasts.quantile(0.95, axis=1)

ax.fill_between(t_fwd, q05, q95, alpha=0.15, color="blue", label="90% interval")
ax.fill_between(t_fwd, q25, q75, alpha=0.3, color="blue", label="50% interval")
ax.plot(t_fwd, q50, "b-", linewidth=1.5, label="Median forecast")

ax.axhline(y=est["mu"], color="red", linestyle="--", linewidth=1, label=f"$\\hat{{\\mu}}={est['mu']:.2f}$")
ax.set_xlabel("Time Step (business days)")
ax.set_ylabel("Spread Level")
ax.set_title("OU Parameter Estimation and Forward Simulation")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
```


## Summary of Stochastic Processes

The following table summarizes the key properties of each process:

| Process | SDE | Mean Reversion | Jumps | Use Case |
|---------|-----|----------------|-------|----------|
| GBM | \(dS = \mu S\,dt + \sigma S\,dW\) | No | No | Stock prices, short-horizon commodity |
| OU | \(dX = \theta(\mu - X)\,dt + \sigma\,dW\) | Yes | No | Spreads, basis, calendar rolls |
| OUJ | \(dX = \theta(\mu - X)\,dt + \sigma\,dW + J\,dN\) | Yes | Yes | Energy spreads with supply shocks |

When choosing a model, consider:

- **GBM** when the price has no natural equilibrium and you are modeling over
  short horizons where mean reversion is negligible.
- **OU** when the price or spread has a clear long-run equilibrium and
  deviations are temporary.
- **OUJ** when the data exhibits both mean reversion and occasional large
  jumps, particularly in energy markets subject to physical supply disruptions.


## References

- Black, F. and Scholes, M. (1973). "The Pricing of Options and
  Corporate Liabilities." *Journal of Political Economy*, 81(3), 637--654.

- Fama, E. F. (1970). "Efficient Capital Markets: A Review of
  Theory and Empirical Work." *Journal of Finance*, 25(2), 383--417.

- Merton, R. C. (1976). "Option Pricing when Underlying Stock
  Returns are Discontinuous." *Journal of Financial Economics*, 3(1--2),
  125--144.

- Uhlenbeck, G. E. and Ornstein, L. S. (1930). "On the
  Theory of the Brownian Motion." *Physical Review*, 36(5), 823--841.

- Vasicek, O. (1977). "An Equilibrium Characterization of the
  Term Structure." *Journal of Financial Economics*, 5(2), 177--188.
