# Fixed Income Analytics

This guide provides a conceptual foundation for fixed income analytics as
implemented in the **risktools** library. We begin with the time value of money,
build up to bond pricing and duration, and then cover net present value and swap
valuation. Each section develops the underlying mathematics before showing how
to apply the corresponding `risktools` function.


## Time Value of Money

The time value of money is the cornerstone of all fixed income analysis. A
dollar received today is worth more than a dollar received in the future because
today's dollar can be invested to earn a return. This principle gives rise to two
fundamental operations: *discounting* (converting future values to present
values) and *compounding* (converting present values to future values).

### Discrete Compounding

Under discrete compounding with an annual interest rate \(r\) and
\(n\) compounding periods, the relationship between present value (PV) and
future value (FV) is:

$$
FV = PV \cdot (1 + r)^{n}
$$

Equivalently, the present value of a future cash flow is obtained by discounting:

$$
PV = \frac{FV}{(1 + r)^{n}}
$$

The quantity \(d(n) = 1 / (1 + r)^{n}\) is called the **discount factor**
for period \(n\). Discount factors are always between zero and one for
positive interest rates, and they decrease as the time horizon lengthens,
reflecting the greater uncertainty and opportunity cost associated with cash
flows that are further away.

When interest is compounded \(m\) times per year at a nominal annual rate
\(r\), the per-period rate is \(r/m\) and the number of periods over
\(T\) years is \(n = m \cdot T\):

$$
PV = \frac{FV}{\left(1 + \dfrac{r}{m}\right)^{m \cdot T}}
$$

### Continuous Compounding

As the compounding frequency \(m\) increases without bound, we arrive at
continuous compounding. The discount factor under continuous compounding is an
exponential function:

$$
PV = FV \cdot e^{-r \, t}
$$

where \(t\) is measured in years. Continuous compounding is the standard
convention for zero-coupon yield curves (including the US Treasury curve used by
`ir_df_us()`) because it simplifies many analytical expressions.

**Example.** A payment of $1,000 due in 3 years, discounted at a continuously
compounded rate of 5%:

$$
PV = 1000 \cdot e^{-0.05 \times 3} = 1000 \cdot e^{-0.15} \approx 860.71
$$

**Reference.** Fabozzi, F.J. (2007). *Fixed Income Analysis*, 2nd ed. CFA
Institute Investment Series. John Wiley & Sons.


## Bond Pricing

A coupon-bearing bond is a security that promises a sequence of periodic coupon
payments and a lump-sum return of face value at maturity. Pricing a bond amounts
to discounting each of these promised cash flows back to the present.

### The Price--Yield Relationship

Consider a bond with face value \(F\), annual coupon rate \(c\),
\(m\) coupon payments per year, \(n = m \cdot T\) total periods, and
yield to maturity \(y\). The coupon payment each period is
\(C = c \cdot F / m\). The price is the sum of the present values of all
cash flows:

$$
P = \sum_{i=1}^{n} \frac{C}{\left(1 + \dfrac{y}{m}\right)^{i}}
  + \frac{F}{\left(1 + \dfrac{y}{m}\right)^{n}}
$$

This can also be written using the annuity formula:

$$
P = C \cdot \frac{1 - \left(1 + \dfrac{y}{m}\right)^{-n}}{\dfrac{y}{m}}
  + \frac{F}{\left(1 + \dfrac{y}{m}\right)^{n}}
$$

### Par, Premium, and Discount Bonds

The relationship between the coupon rate \(c\) and the yield to maturity
\(y\) determines whether a bond trades at, above, or below its face value:

- **Par bond** (\(c = y\)): The bond's price equals its face value,
  \(P = F\).
- **Premium bond** (\(c > y\)): The coupon rate exceeds the required yield,
  so investors pay more than face value, \(P > F\).
- **Discount bond** (\(c < y\)): The coupon rate is below the required
  yield, so the bond sells below face value, \(P < F\).

### Using `bond()`

The `risktools.bond()` function computes the price, cash flow schedule, and
Macaulay duration of a fixed-rate bond. Its parameters map directly to the
notation above:

```python
import risktools as rt

# Price a 5-year, 6% semi-annual coupon bond at a 5% yield
price = rt.bond(ytm=0.05, c=0.06, T=5, m=2, output="price")
print(f"Bond price: ${price:.4f}")

# Retrieve the full cash flow table
df = rt.bond(ytm=0.05, c=0.06, T=5, m=2, output="df")
print(df)

# Compute Macaulay duration
dur = rt.bond(ytm=0.05, c=0.06, T=5, m=2, output="duration")
print(f"Macaulay duration: {dur:.4f} years")
```

The cash flow DataFrame returned by `output="df"` contains the columns
`t_years` (time in years), `cf` (cash flow), `t_periods` (time in coupon
periods), `disc_factor` (discount factor), `pv` (present value of the cash
flow), and `duration` (the contribution of that cash flow to overall Macaulay
duration).

**Verifying par pricing.** When the coupon rate equals the yield to maturity,
the price should equal 100 (per $100 face value):

```python
par_price = rt.bond(ytm=0.05, c=0.05, T=10, m=2, output="price")
print(f"Par bond price: ${par_price:.4f}")  # Should be 100.0000
```

**Reference.** Tuckman, B. and Serrano, A. (2011). *Fixed Income Securities:
Tools for Today's Markets*, 3rd ed. Wiley.


## Duration

Duration measures the sensitivity of a bond's price to changes in interest
rates. The original concept, introduced by Frederick Macaulay in 1938, defines
duration as the weighted-average time to receipt of a bond's cash flows, where
the weights are the present values of those cash flows as a fraction of the
bond's total price.

### Macaulay Duration

Let \(t_i\) denote the time (in years) at which the \(i\)-th cash flow
\(CF_i\) is received, and let \(PV(CF_i)\) denote its present value.
Macaulay duration is:

$$
D_{\text{Mac}} = \frac{1}{P} \sum_{i=1}^{n} t_i \cdot PV(CF_i)
$$

This is a time-weighted average: each future payment date \(t_i\) is
weighted by the fraction of the bond's price attributable to that payment. A
higher duration means the bond's cash flows are, on average, received later.

### Special Cases

- **Zero-coupon bond.** Since there is only one cash flow (the face value at
  maturity), the Macaulay duration of a zero-coupon bond equals its maturity:
  \(D_{\text{Mac}} = T\).

- **Perpetuity.** For a perpetuity (infinite stream of coupons, no principal
  repayment), duration equals \((1 + y) / y\).

- **Coupon bond.** Duration is always less than maturity for a coupon-paying
  bond because the interim coupon payments pull the weighted-average time
  forward.

### Duration as Interest Rate Sensitivity

The practical importance of duration lies in its use as a first-order
approximation for the price change of a bond when yields move. Differentiating
the bond price with respect to the yield gives:

$$
\frac{dP}{P} \approx -D_{\text{Mac}} \cdot \frac{dy}{1 + y/m}
$$

The term \(D_{\text{Mac}} / (1 + y/m)\) is called **modified duration**:

$$
D_{\text{mod}} = \frac{D_{\text{Mac}}}{1 + y/m}
$$

Modified duration directly links a yield change to a percentage price change:

$$
\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y
$$

For example, if a bond has a modified duration of 7.5, then a 1 basis point
(0.01%) increase in yield causes the price to fall by approximately
\(7.5 \times 0.0001 = 0.075\%\).

### Computing Duration with `bond()`

```python
import risktools as rt

# Macaulay duration of a 10-year, 4% semi-annual bond at 5% yield
mac_dur = rt.bond(ytm=0.05, c=0.04, T=10, m=2, output="duration")
print(f"Macaulay duration: {mac_dur:.4f} years")

# Modified duration
mod_dur = mac_dur / (1 + 0.05 / 2)
print(f"Modified duration: {mod_dur:.4f}")

# Approximate price change for a 50 bp rise in yields
price = rt.bond(ytm=0.05, c=0.04, T=10, m=2, output="price")
approx_change = -mod_dur * 0.005 * price
print(f"Approx price change for +50bp: ${approx_change:.4f}")

# Verify against exact repricing
new_price = rt.bond(ytm=0.055, c=0.04, T=10, m=2, output="price")
exact_change = new_price - price
print(f"Exact price change for +50bp:   ${exact_change:.4f}")
```

The small difference between the approximate and exact changes is due to
**convexity** -- the second-order curvature of the price--yield relationship
that duration does not capture.

**Reference.** Macaulay, F.R. (1938). *Some Theoretical Problems Suggested by
the Movements of Interest Rates, Bond Yields and Stock Prices in the United
States since 1856*. National Bureau of Economic Research.


## Net Present Value

Net present value (NPV) extends the discounting framework from bonds to
arbitrary investment projects. An investment that requires an initial outlay and
produces a stream of future cash flows is worth undertaking if the present value
of the inflows exceeds the present value of the outflows.

### Definition

Let \(CF_0\) (typically negative) be the initial investment cost, and let
\(CF_1, CF_2, \ldots, CF_n\) be the subsequent cash flows at times
\(t_1, t_2, \ldots, t_n\). With a discount factor \(d(t_i)\) for each
time, the net present value is:

$$
NPV = \sum_{i=0}^{n} CF_i \cdot d(t_i)
$$

Under continuous compounding with a flat yield \(r\), this becomes:

$$
NPV = CF_0 + \sum_{i=1}^{n} CF_i \cdot e^{-r \, t_i}
$$

The decision rule is straightforward: accept the project if \(NPV > 0\);
reject if \(NPV < 0\). A zero NPV means the project exactly earns the
required rate of return.

### Discount Curve Construction

In practice, interest rates vary by maturity. The `ir_df_us()` function
extracts the current US Treasury zero-coupon yield curve from the Federal
Reserve (via Quandl) and computes discount factors using continuous compounding:

$$
d(t) = e^{-y(t) \cdot t}
$$

where \(y(t)\) is the zero rate for maturity \(t\). The `npv()`
function interpolates this curve using cubic splines to obtain discount factors
at the exact times when cash flows occur.

### Using `npv()`

The `risktools.npv()` function prices a project with evenly spaced periodic
cash flows, an initial cost, and a terminal value. It requires a discount factor
DataFrame from `ir_df_us()`.

```python
import risktools as rt

# Obtain the current US Treasury discount curve
ir = rt.ir_df_us(ir_sens=0.01)

# Project: $375 initial cost, $50 semi-annual cash flows, $250 terminal
# value at year 2
result = rt.npv(
    init_cost=-375,
    C=50,
    cf_freq=0.5,
    F=250,
    T=2,
    disc_factors=ir,
)

print(result)  # DataFrame with columns: t, cf, df, pv
print(f"NPV: ${result.pv.sum():.2f}")
```

### Break-Even Analysis

When the `break_even` flag is set to `True`, the function overrides the
market discount curve with a flat yield specified by `be_yield`. This allows
you to solve for the flat rate at which the project's NPV equals zero -- the
project's internal rate of return.

```python
# Break-even analysis with a flat 3.99% discount rate
result_be = rt.npv(
    init_cost=-375,
    C=50,
    cf_freq=0.5,
    F=250,
    T=2,
    disc_factors=ir,
    break_even=True,
    be_yield=0.0399,
)

print(f"NPV at break-even yield: ${result_be.pv.sum():.2f}")
```

By iterating over different values of `be_yield` and finding the one that
drives the NPV to zero, you can determine the project's internal rate of return
(IRR).

**Reference.** Brealey, R.A., Myers, S.C., and Allen, F. (2020). *Principles
of Corporate Finance*, 13th ed. McGraw-Hill Education.


## Interest Rate Swaps

An interest rate swap (IRS) is a derivative contract in which two parties agree
to exchange interest payments on a notional principal amount. One party pays a
**fixed rate** while the other pays a **floating rate**, typically benchmarked to
a reference rate such as SOFR or the former LIBOR.

### Structure of a Plain Vanilla IRS

In a plain vanilla interest rate swap:

- The **fixed leg** pays a predetermined coupon rate on the notional principal at
  regular intervals (e.g., quarterly).
- The **floating leg** pays a rate determined by the prevailing market rate at
  each reset date, also on the notional principal.
- No principal is exchanged -- only the net difference in interest payments
  changes hands.

The periodic fixed payment on the swap is:

$$
\text{Fixed Payment}_i = N \cdot r_{\text{fix}} \cdot \frac{\Delta t_i}{\text{days in year}}
$$

where \(N\) is the notional, \(r_{\text{fix}}\) is the fixed rate, and
\(\Delta t_i\) is the number of days in period \(i\).

### Valuation

The value of a swap to the **fixed-rate receiver** is:

$$
V_{\text{swap}} = PV(\text{Fixed Leg}) - PV(\text{Floating Leg})
$$

Each leg is valued by discounting its expected cash flows. For the fixed leg,
cash flows are known. For the floating leg, future rates are implied from the
forward curve:

$$
f(t_i, t_{i+1}) = \frac{d(t_i)}{d(t_{i+1})} - 1
$$

where \(d(t)\) denotes the discount factor at time \(t\). The floating
payment for period \([t_i, t_{i+1}]\) is then:

$$
\text{Float Payment}_i = N \cdot f(t_i, t_{i+1})
$$

### Pay vs. Receive Convention

The `pay_rec` parameter controls the sign convention:

- `"rec"` means the user **receives fixed** and pays floating. The value is
  \(PV(\text{Fixed}) - PV(\text{Float})\).
- `"pay"` means the user **pays fixed** and receives floating. The value is
  the negative of the receiver swap.

A swap has zero value at inception when the fixed rate is set equal to the
**par swap rate** -- the rate that equates the present values of the two legs.

### Duration of a Swap

An interest rate swap can be decomposed into a long position in a fixed-rate
bond and a short position in a floating-rate note (or vice versa). Since a
floating-rate note resets to par at each coupon date, its duration is
approximately equal to the time until the next reset. Therefore, the duration
of a receiver swap is approximately equal to the duration of the fixed leg.

The `swap_irs()` function computes this duration as:

$$
D_{\text{swap}} = \sum_{i} \frac{t_i \cdot \text{Net}_i}{PV_{\text{swap}}}
$$

### Discount Curve Construction

The `swap_irs()` function takes discount curves as input via the
`disc_curve` and `float_curve` parameters. These are dictionaries (or
DataFrames) with keys `"times"` (year fractions from today) and
`"discounts"` (corresponding discount factors). The bundled dataset
`usSwapCurves` provides an example curve.

Internally, the function uses cubic spline interpolation
(`scipy.interpolate.splrep`) to obtain discount factors at arbitrary dates,
ensuring smooth pricing even when cash flow dates do not align with the input
curve's grid points.

### Using `swap_irs()`

```python
import risktools as rt

# Load the bundled US swap curve
usSwapCurves = rt.data.open_data("usSwapCurves")

# Price a 2-year receiver swap (receive fixed at 5%, pay floating)
# with quarterly resets on a $1,000,000 notional
result = rt.swap_irs(
    trade_date="2020-01-04",
    eff_date="2020-01-06",
    mat_date="2022-01-06",
    notional=1_000_000,
    pay_rec="rec",
    fixed_rate=0.05,
    float_curve=usSwapCurves,
    reset_freq="Q",
    disc_curve=usSwapCurves,
    days_in_year=360,
    convention="act",
    output="all",
)

print(f"Swap PV:       ${result['pv']:,.2f}")
print(f"Swap Duration: {result['duration']:.4f} years")
print(result["df"])
```

To value a **payer swap** (pay fixed, receive floating), change the convention:

```python
payer_pv = rt.swap_irs(
    trade_date="2020-01-04",
    eff_date="2020-01-06",
    mat_date="2022-01-06",
    notional=1_000_000,
    pay_rec="pay",
    fixed_rate=0.05,
    float_curve=usSwapCurves,
    reset_freq="Q",
    disc_curve=usSwapCurves,
    days_in_year=360,
    convention="act",
    output="price",
)

print(f"Payer swap PV: ${payer_pv:,.2f}")
```

Note that the payer swap PV is the negative of the receiver swap PV, since one
party's asset is the other's liability.

**Reference.** Hull, J.C. (2018). *Options, Futures, and Other Derivatives*,
10th ed. Pearson Education.


## Commodity Swaps

Commodity swaps allow producers and consumers to lock in a fixed price for a
commodity over a specified period. Unlike interest rate swaps, the floating leg
of a commodity swap references the spot or futures price of a physical commodity
(e.g., crude oil, natural gas) rather than an interest rate.

### Calendar Month Average (CMA) Pricing

Many commodity contracts, particularly in crude oil markets, are settled against
a **calendar month average** (CMA) price. The CMA is the arithmetic average of
daily settlement prices over all business days in a given month.

However, complications arise when the underlying futures contract expires
**mid-month** rather than at month end. For example, the CME WTI crude oil
futures contract typically expires around the 20th of the month preceding the
delivery month. This means that for any given calendar month, prices are
determined by two different futures contracts:

1. The **front-month contract** (which expires during the month).
2. The **second-month contract** (which is active after the first contract
   expires).

### Futures Weighting with `swap_fut_weight()`

The `swap_fut_weight()` function computes the weight to apply to each futures
contract when constructing the CMA price. The weighting is based on the number
of business days each contract is active during the month:

$$
w_1 = \frac{\text{Business days up to and including expiry}}
           {\text{Total business days in month}}
$$

$$
w_2 = 1 - w_1
$$

The CMA price is then:

$$
P_{\text{CMA}} = w_1 \cdot F_1 + w_2 \cdot F_2
$$

where \(F_1\) is the price of the first (expiring) contract and \(F_2\)
is the price of the second contract.

```python
import risktools as rt

# Get the first-futures weight for September 2020 WTI
w1 = rt.swap_fut_weight(
    month="2020-09-01",
    contract="cmewti",
    exchange="nymex",
    output="first_fut_weight",
)
print(f"First futures weight:  {w1:.4f}")
print(f"Second futures weight: {1 - w1:.4f}")

# Number of business days for each contract
days1 = rt.swap_fut_weight(
    month="2020-09-01",
    contract="cmewti",
    exchange="nymex",
    output="num_days_fut1",
)
days2 = rt.swap_fut_weight(
    month="2020-09-01",
    contract="cmewti",
    exchange="nymex",
    output="num_days_fut2",
)
print(f"Business days (contract 1): {days1}")
print(f"Business days (contract 2): {days2}")
```

The function accounts for exchange-specific holidays (e.g., NYMEX holidays) when
computing business days, ensuring accurate weighting.

### Pricing a Commodity Swap with `swap_com()`

The `swap_com()` function takes a DataFrame of daily futures settlement prices
for two contracts and computes the weighted-average swap price. It uses the
expiry table and holiday calendar bundled with `risktools` to determine the
correct weighting automatically.

```python
import risktools as rt

# Assume df is a wide DataFrame of daily settlement prices
# with columns "CL0M" and "CL0N" (two consecutive WTI contracts)
# obtained via rt.get_prices() or another data source.

swap_prices = rt.swap_com(
    df=df,
    futures_names=["CL0M", "CL0N"],
    start_dt="2020-05-01",
    end_dt="2020-05-30",
    cmdty="cmewti",
    exchange="nymex",
)

print(swap_prices)
```

The returned DataFrame contains the original two contract prices plus a
`swap` column with the weighted-average price for each date.

The logic is straightforward:

1. Determine the expiry date of the first contract within the pricing window.
2. For dates up to and including the expiry, assign a weight of 1 to the first
   contract.
3. After expiry, assign a weight of 1 to the second contract.
4. Compute the ratio of business days in each regime to get the overall weights.
5. The swap price on each date is:
   \(P_{\text{swap}} = w_1 \cdot F_1 + w_2 \cdot F_2\).


## References

- Brealey, R.A., Myers, S.C., and Allen, F. (2020). *Principles
  of Corporate Finance*, 13th ed. McGraw-Hill Education.

- Fabozzi, F.J. (2007). *Fixed Income Analysis*, 2nd ed. CFA
  Institute Investment Series. John Wiley & Sons.

- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*,
  10th ed. Pearson Education.

- Macaulay, F.R. (1938). *Some Theoretical Problems Suggested
  by the Movements of Interest Rates, Bond Yields and Stock Prices in the
  United States since 1856*. National Bureau of Economic Research.

- Tuckman, B. and Serrano, A. (2011). *Fixed Income Securities:
  Tools for Today's Markets*, 3rd ed. Wiley.
