# Performance Analytics: A Conceptual Guide

This guide provides the theoretical foundations behind the performance analytics
functions in risktools. It is written for students of quantitative finance
who want to understand not just *how* to call a function, but *why* the
mathematics works the way it does. Each section presents definitions, formulas,
intuition, and working code examples.

Throughout this guide, we assume that returns are observed at regular intervals
(daily, weekly, monthly, quarterly, or annually) and are stored in a pandas
`Series` or `DataFrame` with a `DatetimeIndex` whose `freq` attribute is
set. The library uses the following annualization scale factors by default:

| Frequency | Scale |
|---|---|
| Daily / Business-day (`D`, `B`) | 252 |
| Weekly (`W`) | 52 |
| Monthly (`M`, `ME`, `MS`) | 12 |
| Quarterly (`Q`, `QE`, `QS`) | 4 |
| Annual (`Y`, `YE`, `YS`, `A`) | 1 |

---

## 1. Returns

The most elementary concept in performance measurement is the *return* on an
investment. Given an asset price \(P_t\) at time \(t\), there are three
common ways to express the return over a single period.

### Arithmetic (Simple) Return

$$
R_t^{\text{arith}} = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

This is the percentage change in price. It is additive across assets in a
portfolio (the portfolio return is the weighted sum of asset returns) but it does
**not** compound correctly across time.

### Geometric (Compound) Return

When we speak of a *geometric* return over multiple periods, we mean the return
that, if earned every period, would reproduce the same terminal wealth. For a
single period, the geometric return equals the arithmetic return. The distinction
becomes important when we chain returns over time (see Section 2).

### Logarithmic (Continuously Compounded) Return

$$
r_t^{\log} = \ln\!\left(\frac{P_t}{P_{t-1}}\right) = \ln(1 + R_t^{\text{arith}})
$$

Log returns are additive across time:

$$
r_{1 \to n}^{\log} = \sum_{t=1}^{n} r_t^{\log}
$$

This property makes them analytically convenient in models that assume normally
distributed returns (e.g., geometric Brownian motion). However, log returns are
**not** additive across assets in a portfolio, which is why most practitioners
work with arithmetic returns and apply geometric compounding when aggregating
over time.

### Why Geometric Compounding Matters

Consider an investment that gains 50% in one period and then loses 50% in the
next. The arithmetic average return is 0%, suggesting the investor broke even.
In reality:

$$
(1 + 0.50)(1 - 0.50) - 1 = 0.75 - 1 = -0.25
$$

The investor lost 25% of their wealth. The geometric average correctly captures
this compounding effect and is therefore the appropriate measure for evaluating
investment performance over time.

!!! info "Reference"

    Bacon, C. (2004). *Practical Portfolio Performance Measurement and
    Attribution*. Wiley. Chapter 2.

---

## 2. Cumulative Returns

A cumulative return answers the question: "What was my total return from the
beginning of the observation period to the end?"

### Geometric Cumulative Return

The geometrically compounded cumulative return is:

$$
R_{\text{cum}}^{\text{geo}} = \prod_{t=1}^{n}(1 + R_t) - 1
$$

This accounts for the reinvestment of gains and the erosion caused by losses.
Each period's return is applied to the cumulated wealth, not just to the
original investment.

### Arithmetic Cumulative Return

The simple (arithmetic) cumulative return is:

$$
R_{\text{cum}}^{\text{arith}} = \sum_{t=1}^{n} R_t
$$

This is a less common measure that assumes no compounding -- each period's
return is computed on the original capital. It is a meaningful calculation only
when capital is withdrawn and re-invested at the start of each period (e.g.,
certain managed account structures).

### Code Example

The function `risktools.return_cumulative()` computes both variants:

```python
import pandas as pd
import numpy as np
import risktools as rt

# Simulated monthly returns
dates = pd.date_range("2020-01-31", periods=12, freq="ME")
returns = pd.Series(
    [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.005, 0.025, 0.01, -0.03, 0.02],
    index=dates
)

# Geometric cumulative return (default)
geo_cum = rt.return_cumulative(returns, geometric=True)
print(f"Geometric cumulative return: {geo_cum:.4%}")

# Arithmetic cumulative return
arith_cum = rt.return_cumulative(returns, geometric=False)
print(f"Arithmetic cumulative return: {arith_cum:.4%}")
```

Internally, the geometric calculation executes `r.add(1).prod() - 1` and the
arithmetic calculation executes `r.sum()`. When passed a `DataFrame`, the
function applies the computation column-wise.

---

## 3. Annualized Returns

Cumulative returns are difficult to compare across investments that have
different observation histories. An annualized return rescales a cumulative
return to an annual equivalent, allowing apples-to-apples comparisons regardless
of the measurement period.

### Geometric Annualized Return

$$
R_{\text{ann}}^{\text{geo}}
= \left[\prod_{t=1}^{n}(1 + R_t)\right]^{\frac{s}{n}} - 1
= (1 + R_{\text{cum}}^{\text{geo}})^{\frac{s}{n}} - 1
$$

where:

- \(n\) is the total number of observed periods, and
- \(s\) is the annualization scale (the number of periods per year).

The exponent \(s/n\) converts the cumulative growth factor to an annual
growth factor. In effect, we are computing the constant annual rate that, over
the same duration, would have produced the same cumulative wealth.

### Arithmetic Annualized Return

$$
R_{\text{ann}}^{\text{arith}} = \bar{R} \cdot s
$$

where \(\bar{R}\) is the arithmetic mean of the period returns. This simply
scales the average period return to an annual figure. It overstates the true
growth rate when returns are volatile because it ignores the compounding drag
(variance drain).

### Code Example

`risktools.return_annualized()` implements both approaches:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2020-01-31", periods=36, freq="ME")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.008, 0.04, 36), index=dates)

ann_geo = rt.return_annualized(returns, geometric=True)
print(f"Annualized return (geometric): {ann_geo:.4%}")

ann_arith = rt.return_annualized(returns, geometric=False)
print(f"Annualized return (arithmetic): {ann_arith:.4%}")
```

When no `scale` argument is provided, the function infers the scale from the
index frequency. You can override this by passing an integer explicitly.

!!! info "References"

    Bacon, C. (2004). *Practical Portfolio Performance Measurement and
    Attribution*. Wiley. p. 6.

    CFA Institute. (2020). *Global Investment Performance Standards (GIPS)*. The
    GIPS standards require geometric (time-weighted) return computation for
    performance reporting.

---

## 4. Annualized Standard Deviation

Standard deviation is the most widely used measure of total risk. For a series
of period returns \(R_1, R_2, \ldots, R_n\):

$$
\sigma = \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(R_t - \bar{R})^2}
$$

### The Square-Root-of-Time Rule

To express volatility on an annual basis, practitioners apply the
**square-root-of-time rule**:

$$
\sigma_{\text{ann}} = \sigma_{\text{period}} \cdot \sqrt{s}
$$

where \(s\) is the annualization scale factor. The theoretical justification
rests on the assumption that returns are **independent and identically
distributed (i.i.d.)**. Under this assumption, the variance of a sum of
\(s\) independent random variables equals \(s\) times the variance of
one variable:

$$
\text{Var}\!\left(\sum_{t=1}^{s} R_t\right)
= s \cdot \text{Var}(R_t)
\quad\Longrightarrow\quad
\text{SD}\!\left(\sum_{t=1}^{s} R_t\right)
= \sqrt{s}\;\cdot\;\text{SD}(R_t)
$$

**Caveats.** In practice, asset returns exhibit serial correlation (momentum or
mean reversion) and time-varying volatility (volatility clustering). When
returns are positively autocorrelated, the square-root rule *understates* true
annual risk; when negatively autocorrelated, it *overstates* it. The rule
remains ubiquitous because of its simplicity and because deviations from i.i.d.
are difficult to estimate reliably with short sample histories.

### Code Example

`risktools.sd_annualized()` applies this rule:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2020-01-02", periods=504, freq="B")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.0004, 0.012, 504), index=dates)

ann_vol = rt.sd_annualized(returns)
print(f"Annualized volatility: {ann_vol:.4%}")

# Verify manually: daily std * sqrt(252)
manual = returns.std() * np.sqrt(252)
print(f"Manual calculation:    {manual:.4%}")
```

!!! info "Reference"

    Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.).
    Pearson. Chapter 15 -- discussion of the square-root-of-time rule for
    scaling volatility.

---

## 5. Sharpe Ratio

The Sharpe Ratio, introduced by William Sharpe in 1966, is the foundational
risk-adjusted performance measure. It answers the question: "How much excess
return did the investor earn per unit of total risk?"

### Definition

$$
\text{SR} = \frac{R_p - R_f}{\sigma_p}
$$

where:

- \(R_p\) is the portfolio return,
- \(R_f\) is the risk-free rate, and
- \(\sigma_p\) is the standard deviation of portfolio returns.

A higher Sharpe Ratio indicates a more efficient risk-return trade-off.

### Annualized Sharpe Ratio

To compare strategies measured at different frequencies, we annualize both the
numerator and denominator. The annualized Sharpe Ratio is:

$$
\text{SR}_{\text{ann}}
= \frac{R_{\text{ann}}^{\text{excess}}}{\sigma_{\text{ann}}}
= \frac{\left[\prod_{t=1}^{n}(1 + R_t - R_f)\right]^{s/n} - 1}
       {\sigma_{\text{period}} \cdot \sqrt{s}}
$$

When `geometric=False`, the numerator uses the arithmetic annualized return
instead.

### Limitations

The Sharpe Ratio has several well-known limitations:

1. **Assumes normally distributed returns.** When returns are skewed or have
   fat tails, standard deviation does not fully capture risk.
2. **Penalizes upside volatility.** An asset with large positive outliers will
   have a high standard deviation, lowering its Sharpe Ratio despite favorable
   returns.
3. **Sensitive to the measurement period.** The choice of time frame can
   significantly alter the result.
4. **Assumes a constant risk-free rate.** In practice, the risk-free rate
   changes over time.

These shortcomings motivated the development of alternative measures such as the
Sortino Ratio, the Omega Ratio, and downside risk metrics.

### Code Example

`risktools.sharpe_ratio_annualized()` computes the annualized version:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2018-01-31", periods=60, freq="ME")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.008, 0.04, 60), index=dates)

sr = rt.sharpe_ratio_annualized(returns, Rf=0.001, geometric=True)
print(f"Annualized Sharpe Ratio: {sr:.4f}")
```

!!! info "Reference"

    Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business*,
    39(1), 119--138.

---

## 6. Omega-Sharpe Ratio

The Omega-Sharpe Ratio reformulates the Omega Ratio (Keating and Shadwick, 2002)
into a ranking statistic that is structurally comparable to the Sharpe Ratio.

### Omega Ratio -- Background

The Omega Ratio partitions the return distribution at a threshold
\(\tau\) (the Minimum Acceptable Return, or MAR) and compares the
probability-weighted gains above \(\tau\) to the probability-weighted
losses below it:

$$
\Omega(\tau)
= \frac{\int_{\tau}^{\infty} [1 - F(r)]\,dr}{\int_{-\infty}^{\tau} F(r)\,dr}
= \frac{\text{Upside Potential}}{\text{Downside Potential}}
$$

where \(F(r)\) is the cumulative distribution function of returns.

### Omega-Sharpe Ratio

The Omega-Sharpe Ratio converts this to a Sharpe-like form by subtracting one:

$$
\text{OmegaSharpe}(R, \tau)
= \frac{\text{UpsidePotential}(R, \tau) - \text{DownsidePotential}(R, \tau)}
       {\text{DownsidePotential}(R, \tau)}
= \Omega(\tau) - 1
$$

where the sample-based upside and downside potentials are:

$$
\text{UpsidePotential}(R, \tau)
= \frac{1}{n}\sum_{t=1}^{n} \max(R_t - \tau,\; 0)
$$

$$
\text{DownsidePotential}(R, \tau)
= \frac{1}{n}\sum_{t=1}^{n} \max(\tau - R_t,\; 0)
$$

Unlike the Sharpe Ratio, the Omega-Sharpe Ratio considers the *entire* return
distribution -- all moments (mean, variance, skewness, kurtosis, etc.) are
implicitly incorporated. It makes no assumption of normality.

### Code Example

`risktools.omega_sharpe_ratio()` computes this measure:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2018-01-31", periods=60, freq="ME")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.008, 0.04, 60), index=dates)

mar = 0.005  # Monthly MAR of 0.5%
osr = rt.omega_sharpe_ratio(returns, MAR=mar)
print(f"Omega-Sharpe Ratio: {osr:.4f}")
```

!!! info "Reference"

    Keating, C. and Shadwick, W. F. (2002). A Universal Performance Measure.
    *Journal of Performance Measurement*, 6(3), 59--84.

---

## 7. Downside Risk Measures

Traditional volatility (standard deviation) treats upside and downside
dispersion symmetrically. Investors, however, are generally concerned only
with outcomes that fall below some target. Downside risk measures address this
asymmetry by focusing exclusively on the unfavorable portion of the return
distribution.

### Minimum Acceptable Return (MAR)

All downside measures require a threshold, often called the **Minimum
Acceptable Return (MAR)** or target return \(\tau\). Common choices include:

- \(\tau = 0\) (absolute loss avoidance),
- \(\tau = R_f\) (the risk-free rate), or
- \(\tau = \bar{R}\) (the mean return, yielding the semi-deviation).

### Downside Deviation

Downside deviation captures the root-mean-square shortfall below the MAR:

$$
\text{DD}(R, \tau)
= \sqrt{\frac{1}{n}\sum_{t=1}^{n}\left[\min(R_t - \tau,\;0)\right]^2}
= \sqrt{\frac{1}{n}\sum_{t=1}^{n}\left[\max(\tau - R_t,\;0)\right]^2}
$$

By using \(n\) (the full number of observations) in the denominator rather
than the count of below-target returns, we avoid overstating the risk of
distributions with few but extreme shortfalls. The `method` parameter controls
this choice: `"full"` (default) uses \(n\); `"subset"` uses the count
of below-MAR observations only.

**Semi-deviation** is the special case where \(\tau = \bar{R}\):

```python
import risktools as rt

# Semi-deviation: set MAR equal to the mean return
semi_dev = rt.downside_deviation(returns, MAR=returns.mean())
```

### Downside Potential

Downside potential is the first lower partial moment -- the average shortfall
below the MAR:

$$
\text{DP}(R, \tau) = \frac{1}{n}\sum_{t=1}^{n}\max(\tau - R_t,\;0)
$$

This is obtained by passing `potential=True` to
`risktools.downside_deviation()`.

### Upside Risk and Upside Potential

The mirror images of the downside measures capture the favorable part of the
distribution:

$$
\text{UpsideRisk}(R, \tau) = \sqrt{\frac{1}{n}\sum_{t=1}^{n}[\max(R_t - \tau,\;0)]^2}
$$

$$
\text{UpsidePotential}(R, \tau) = \frac{1}{n}\sum_{t=1}^{n}\max(R_t - \tau,\;0)
$$

`risktools.upside_risk()` computes these, controlled by the `stat`
parameter: `"risk"` for upside risk (default), `"variance"` for the squared
version, and `"potential"` for upside potential.

### Code Example

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2018-01-31", periods=60, freq="ME")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.008, 0.04, 60), index=dates)

mar = 0.0

dd = rt.downside_deviation(returns, MAR=mar)
print(f"Downside deviation:  {dd:.6f}")

dp = rt.downside_deviation(returns, MAR=mar, potential=True)
print(f"Downside potential:  {dp:.6f}")

ur = rt.upside_risk(returns, MAR=mar, stat="risk")
print(f"Upside risk:         {ur:.6f}")

up = rt.upside_risk(returns, MAR=mar, stat="potential")
print(f"Upside potential:    {up:.6f}")
```

!!! info "Reference"

    Sortino, F. A. and van der Meer, R. (1991). Downside Risk. *Journal of
    Portfolio Management*, 17(4), 27--31.

---

## 8. Drawdowns

A **drawdown** measures the decline from a historical peak in cumulative
wealth. It is one of the most intuitive risk measures because it directly
answers the question investors care about most: "How much could I have lost
from peak to trough?"

### Definition

Let \(W_t\) denote the cumulative wealth index at time \(t\):

$$
W_t = \prod_{i=1}^{t}(1 + R_i)
$$

The running maximum (high-water mark) is:

$$
M_t = \max_{0 \le i \le t} W_i
$$

The drawdown at time \(t\) is:

$$
D_t = \frac{W_t}{M_t} - 1
$$

Note that \(D_t \le 0\) during drawdown periods and \(D_t = 0\) at new
highs.

For arithmetic (non-compounded) returns, the wealth index is instead
\(W_t = 1 + \sum_{i=1}^{t} R_i\).

### Maximum Drawdown

The **maximum drawdown** over the observation period is:

$$
\text{MaxDD} = \min_{t}\; D_t
$$

This represents the worst peak-to-trough decline experienced.

### Drawdown Anatomy

Each drawdown episode has three phases:

1. **Peak-to-trough**: the period from the high-water mark to the lowest point
   of the drawdown.
2. **Recovery**: the period from the trough back to a new high-water mark.
3. **Length**: the total duration from the beginning of the decline to full
   recovery (or the end of the sample if recovery has not occurred).

### Code Example

`risktools.drawdowns()` computes the drawdown time series, and
`risktools.find_drawdowns()` identifies individual drawdown episodes with
their start, trough, end, and durations:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2018-01-02", periods=504, freq="B")
np.random.seed(42)
returns = pd.Series(np.random.normal(0.0003, 0.012, 504), index=dates)

# Drawdown time series
dd_series = rt.drawdowns(returns, geometric=True)

# Find individual drawdown episodes
dd_info = rt.find_drawdowns(returns, geometric=True)

# Maximum drawdown
max_dd = dd_info["return"].min()
print(f"Maximum drawdown: {max_dd:.4%}")

# Longest drawdown
longest = dd_info["length"].max()
print(f"Longest drawdown (periods): {longest}")

# Longest recovery
longest_recovery = dd_info["recovery"].max()
print(f"Longest recovery (periods): {longest_recovery}")
```

The `find_drawdowns` function returns a dictionary with the following keys:

- `"return"`: the depth (minimum return) of each drawdown episode.
- `"from"`: the index position where each drawdown begins.
- `"trough"`: the index position of the deepest point.
- `"to"`: the index position where the drawdown ends (recovery or end of sample).
- `"length"`: total length of each episode.
- `"peaktotrough"`: number of periods from start to trough.
- `"recovery"`: number of periods from trough to recovery.

!!! info "Reference"

    Bacon, C. (2004). *Practical Portfolio Performance Measurement and
    Attribution*. Wiley. p. 88.

---

## 9. CAPM Beta

The Capital Asset Pricing Model (CAPM), developed by Sharpe (1964), Lintner
(1965), and Mossin (1966), establishes a linear relationship between an asset's
expected excess return and the excess return on the market portfolio.

### The Single-Factor Model

$$
R_a - R_f = \alpha + \beta\,(R_b - R_f) + \epsilon
$$

where:

- \(R_a\) is the asset return,
- \(R_b\) is the benchmark (market) return,
- \(R_f\) is the risk-free rate,
- \(\alpha\) is the intercept (Jensen's alpha -- the risk-adjusted excess return),
- \(\beta\) is the slope coefficient, and
- \(\epsilon\) is the idiosyncratic error term.

The beta coefficient measures the **systematic risk** of the asset relative to
the benchmark:

$$
\beta_{a,b}
= \frac{\text{Cov}(R_a - R_f,\; R_b - R_f)}{\text{Var}(R_b - R_f)}
= \frac{\sum_{t=1}^{n}(R_{a,t} - \bar{R}_a)(R_{b,t} - \bar{R}_b)}
       {\sum_{t=1}^{n}(R_{b,t} - \bar{R}_b)^2}
$$

Interpretation:

- \(\beta = 1\): the asset moves in lockstep with the benchmark.
- \(\beta > 1\): the asset amplifies benchmark movements (higher systematic risk).
- \(\beta < 1\): the asset dampens benchmark movements (lower systematic risk).
- \(\beta < 0\): the asset moves inversely to the benchmark (a hedge).

### Bull Beta and Bear Beta

The overall beta may mask asymmetric behavior in up and down markets. **Bull
beta** (\(\beta^{+}\)) is estimated using only periods where the benchmark
had positive excess returns. **Bear beta** (\(\beta^{-}\)) is estimated
using only periods where the benchmark had negative excess returns.

$$
\beta^{+}: \quad R_{a,t} - R_f = \alpha^{+} + \beta^{+}(R_{b,t} - R_f) + \epsilon_t
\qquad \text{for } t \text{ where } R_{b,t} - R_f > 0
$$

$$
\beta^{-}: \quad R_{a,t} - R_f = \alpha^{-} + \beta^{-}(R_{b,t} - R_f) + \epsilon_t
\qquad \text{for } t \text{ where } R_{b,t} - R_f < 0
$$

An investor prefers a manager with \(\beta^{+} > \beta^{-}\): the manager
captures more of the upside than the downside.

### Code Example

`risktools.capm_beta()` performs the regression using scikit-learn's
`LinearRegression`:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2015-01-02", periods=756, freq="B")
np.random.seed(42)

# Simulate a benchmark and a correlated asset
market = pd.Series(np.random.normal(0.0004, 0.01, 756), index=dates)
asset = 1.2 * market + pd.Series(
    np.random.normal(0.0001, 0.005, 756), index=dates
)

beta_all = rt.capm_beta(asset, market, Rf=0, kind="all")
beta_bull = rt.capm_beta(asset, market, Rf=0, kind="bull")
beta_bear = rt.capm_beta(asset, market, Rf=0, kind="bear")

print(f"Overall beta: {beta_all:.4f}")
print(f"Bull beta:    {beta_bull:.4f}")
print(f"Bear beta:    {beta_bear:.4f}")
```

When `Ra` is a `DataFrame`, the function computes beta for each column
(asset) against the common benchmark `Rb`, returning a `Series` of betas.

!!! info "Reference"

    Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium
    under Conditions of Risk. *Journal of Finance*, 19(3), 425--442.

---

## 10. Timing Ratio

The **Timing Ratio** uses bull and bear betas to evaluate whether a portfolio
manager demonstrates skill in market timing -- the ability to increase
exposure before market advances and decrease exposure before market declines.

### Definition

$$
\text{TimingRatio} = \frac{\beta^{+}}{\beta^{-}}
$$

Interpretation:

- **TimingRatio > 1**: the manager has higher sensitivity to rising markets
  than to falling markets. This is consistent with successful market timing --
  the manager increases exposure when the market goes up and reduces exposure
  when it goes down.
- **TimingRatio = 1**: the manager's sensitivity is symmetric; no evidence of
  timing skill.
- **TimingRatio < 1**: the manager has *perverse* timing -- higher exposure to
  down markets than to up markets, which destroys value.

### Connection to the Henriksson-Merton Framework

Henriksson and Merton (1981) proposed a formal test for market timing ability
based on the following regression:

$$
R_a - R_f = \alpha + \beta_1 (R_b - R_f)
+ \beta_2 \max(0,\; -(R_b - R_f)) + \epsilon
$$

The coefficient \(\beta_2\) captures the incremental response to negative
market returns. A positive \(\beta_2\) indicates that the manager
successfully reduces exposure in down markets. The Timing Ratio provides a
simpler, ratio-based diagnostic that captures the same intuition: a ratio
greater than one implies that the manager participates more in up markets than
in down markets.

### Code Example

`risktools.timing_ratio()` computes this directly:

```python
import pandas as pd
import numpy as np
import risktools as rt

dates = pd.date_range("2015-01-02", periods=756, freq="B")
np.random.seed(42)

market = pd.Series(np.random.normal(0.0004, 0.01, 756), index=dates)
asset = 1.2 * market + pd.Series(
    np.random.normal(0.0001, 0.005, 756), index=dates
)

tr = rt.timing_ratio(asset, market, Rf=0)
print(f"Timing Ratio: {tr:.4f}")
```

!!! info "Reference"

    Henriksson, R. D. and Merton, R. C. (1981). On Market Timing and Investment
    Performance. II. Statistical Procedures for Evaluating Forecasting Skills.
    *Journal of Business*, 54(4), 513--533.

---

## Summary of Functions

The following table maps the concepts discussed in this guide to the
corresponding `risktools` functions:

| Concept | Function | Section |
|---|---|---|
| Cumulative return | `return_cumulative()` | 2 |
| Annualized return | `return_annualized()` | 3 |
| Excess return | `return_excess()` | 5 |
| Annualized standard deviation | `sd_annualized()` | 4 |
| Annualized Sharpe Ratio | `sharpe_ratio_annualized()` | 5 |
| Omega-Sharpe Ratio | `omega_sharpe_ratio()` | 6 |
| Downside deviation | `downside_deviation()` | 7 |
| Upside risk / potential | `upside_risk()` | 7 |
| Drawdown levels | `drawdowns()` | 8 |
| Drawdown episodes | `find_drawdowns()` | 8 |
| CAPM beta | `capm_beta()` | 9 |
| Timing ratio | `timing_ratio()` | 10 |

---

## References

- Bacon, C. (2004). *Practical Portfolio Performance Measurement
  and Attribution*. Wiley.

- CFA Institute. (2020). *Global Investment Performance Standards
  (GIPS)*. CFA Institute.

- Henriksson, R. D. and Merton, R. C. (1981). On Market
  Timing and Investment Performance. II. Statistical Procedures for Evaluating
  Forecasting Skills. *Journal of Business*, 54(4), 513--533.

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
  (10th ed.). Pearson.

- Keating, C. and Shadwick, W. F. (2002). A Universal
  Performance Measure. *Journal of Performance Measurement*, 6(3), 59--84.

- Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market
  Equilibrium under Conditions of Risk. *Journal of Finance*, 19(3), 425--442.

- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of
  Business*, 39(1), 119--138.

- Sortino, F. A. and van der Meer, R. (1991). Downside Risk.
  *Journal of Portfolio Management*, 17(4), 27--31.
