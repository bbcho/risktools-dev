# Data Reference

risktools ships with over 20 curated datasets for commodity trading analytics,
accessible through the `risktools.data.open_data()` function. These datasets
are stored as JSON and CSV files within the package.

```python
import risktools as rt

# List all available datasets
rt.data.get_names()

# Load a dataset
df = rt.data.open_data('dfwide')
```

## Futures Prices

### dfwide

Wide-format DataFrame of daily continuous futures settlement prices with a
`DatetimeIndex`. Columns are contract codes (e.g., `CL01` for WTI front
month, `CL12` for 12th month, `NG01` for natural gas front month).

Contracts included: CL (WTI crude), HO (heating oil / ULSD), RB (RBOB gasoline),
NG (natural gas), BRN (ICE Brent).

```python
df = rt.data.open_data('dfwide')
# DatetimeIndex, ~40 columns of futures prices
```

### dflong

Long-format (stacked) version of the same futures price data. Returned as a
pandas Series with a MultiIndex of (date, contract).

```python
df = rt.data.open_data('dflong')
# Access a single contract
cl01 = df['CL01']
```

### ohlc

Open-High-Low-Close price data for a commodity contract. Useful for
candlestick charting and volatility analysis.

### eurodollar

Eurodollar futures price data used for interest rate curve construction.

## Expiry and Calendar Data

### expiry_table

Historical and forward futures contract expiry dates. Key columns:

- `Last_Trade` -- last trading date for the contract
- `cmdty` -- commodity code (e.g., `cmewti`, `cmeulsd`, `cmeng`)
- `First_Trade` -- first trading date

Used by `swap_com()` and `swap_fut_weight()` to calculate calendar month
average weights.

```python
exp = rt.data.open_data('expiry_table')
# Filter for WTI
wti_exp = exp[exp.cmdty == 'cmewti']
```

### holidaysOil

Oil market holiday calendars for NYMEX and ICE exchanges.

- `key` -- exchange name (`nymex` or `ice`)
- `value` -- holiday date

### tradeCycle

Trade cycle data for commodity markets.

### tradeHubs

Geographic trading hub information for commodity markets.

### tradeprocess

Trade process workflow data.

## Interest Rate and Swap Data

### usSwapCurves

Sample output of the R package RQuantlib's `DiscountCurve()` function,
stored as a dictionary with keys:

- `times` -- numpy array of year fractions (0, 0.0833, ..., 30)
- `discounts` -- numpy array of discount factors
- `forwards` -- numpy array of forward rates
- `zerorates` -- numpy array of zero coupon rates
- `flatQuotes` -- list with a single flat quote value
- `params` -- dictionary of curve parameters
- `table` -- DataFrame summary

Used as input to `swap_irs()` for interest rate swap pricing.

```python
curves = rt.data.open_data('usSwapCurves')
pv = rt.swap_irs(
    trade_date='2020-01-04', eff_date='2020-01-06', mat_date='2022-01-06',
    float_curve=curves, disc_curve=curves, output='price'
)
```

### usSwapCurvesPar

Par swap curve version of `usSwapCurves`. Same dictionary structure.

### tsQuotes

Term structure quote data for swap curve construction.

### wti_swap

Sample WTI swap pricing data.

## Crude Oil Data

### crudeOil

Comprehensive crude oil dataset returned as a dictionary:

- `crudes` -- crude oil specifications and properties
- `CanadianAssays` -- Canadian crude assay data from CrudeMonitor
- `bpAssays` -- BP publicly available crude assays
- `xomAssays` -- ExxonMobil publicly available crude assays
- `CanadaPrices` -- Canadian crude price differentials

### cushing

Cushing, Oklahoma storage and pricing data (dictionary):

- `c1` -- front month WTI prices
- `c2` -- second month WTI prices
- `c1c2` -- calendar spread (c1 - c2)
- `storage` -- Cushing storage levels

### futuresRef

Futures contract reference data (dictionary):

- `ContractMonths` -- which months each commodity trades
- `Specifications` -- contract specifications (tick size, units, etc.)

### fizdiffs

FIZ (Fuel, Industrial, Zinc) differential data for refined products pricing.

## EIA Data

### eiaStocks

Sample Energy Information Administration (EIA) weekly stocks data.
Columns: `date`, `value`, `series`.

Series include: `NGLower48` (natural gas storage), crude oil stocks by PADD, and
refined product inventories.

```python
stocks = rt.data.open_data('eiaStocks')
ng = stocks[stocks.series == 'NGLower48']
```

### eiaStorageCap

EIA crude oil storage capacity by PADD region.

### tickers_eia

Mapping table of EIA API ticker codes to commodity categories. Columns:

- `tick_eia` -- EIA series ID
- `sd_category` -- supply/demand category (`mogas`, `diesel`, `jet`, `resid`)
- `category` -- sub-category (`production`, `imports`, `stocks`, etc.)

Used by `chart_eia_sd()` to build supply/demand balance charts.

## FX Data

### fxfwd

Foreign exchange forward curve data (dictionary):

- `historical` -- historical FX rates
- `curve` -- current forward curve

## Equity Data

### stocks

Sample equity price data (dictionary):

- `spy` -- SPDR S&P 500 ETF
- `uso` -- United States Oil Fund
- `ry` -- Royal Bank of Canada

### planets

Planetary data (used for examples and testing).

## Refinery Optimization Data

### refineryLPdata

Input data for the refinery LP optimizer (dictionary):

- `inputs` -- crude input costs with columns: `info`, `LightSweet`, `HeavySour`
- `outputs` -- product yields and constraints with columns: `product`, `prices`,
  `max_prod`, `LightSweet_yield`, `HeavySour_yield`

```python
data = rt.data.open_data('refineryLPdata')
result = rt.refinery_lp(data['inputs'], data['outputs'])
```

## API Reference

See [Data API](api/data.md) for full function documentation.
