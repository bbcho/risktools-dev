# RiskTools Code Review

**Date:** 2026-02-07
**Version:** 0.2.8.7
**Reviewer:** Claude (automated review)

---

## Executive Summary

RiskTools is a Python port of the R `RTL` (Risk Tools Library) package, providing quantitative finance and commodity trading analytics. The codebase is approximately 6,400 lines of Python across 10 modules, with Cython extensions for performance-critical simulation code. It covers stochastic simulations (GBM, OU, OUJ), performance analytics, swap pricing, charting, and data API integrations.

Overall, the library provides substantial domain-specific functionality. However, there are several categories of issues that should be addressed, ranging from bugs to security concerns to maintainability improvements.

---

## Critical Issues

### 1. Bug: `npv()` copies `None` before checking it (`_main_functions.py:633-636`)

```python
def npv(..., disc_factors=None, ...):
    disc_factors = disc_factors.copy()  # AttributeError if None
    if disc_factors is None:             # Dead code - never reached
        raise ValueError(...)
```

The `disc_factors.copy()` call on line 633 will raise `AttributeError: 'NoneType' object has no attribute 'copy'` if `disc_factors` is `None`. The `None` check on line 635 is dead code. The check must come before the copy.

### 2. Bug: `prompt_beta()` chart labels "bear" twice (`_main_functions.py:579`)

```python
fig.add_trace(
    _go.Scatter(x=out.index, y=out["bull"], mode="lines", name="bear")  # should be "bull"
)
```

The third trace for bull betas is mislabeled as "bear" in the legend.

### 3. Bug: Missing `ValueError` raise in `_MVSIM.plot_portfolio()` (`_multivariate.py:959`)

```python
if self._frontier is None:
    ValueError("plot_efficient_frontier method must be run prior to this method")
```

The `ValueError` is constructed but never raised. Should be `raise ValueError(...)`.

### 4. Bug: Bitwise `~` used instead of logical `not` on non-numpy types (`_multivariate.py:133,207-213`)

```python
if ~isinstance(cor, _np.ndarray):  # Always True! ~ on bool gives -1/-2, both truthy
    cor = _np.array(cor)
```

Throughout `_multivariate.py` and `_pa.py`, the bitwise NOT operator `~` is used with `isinstance()`. For Python booleans, `~True` is `-2` (truthy) and `~False` is `-1` (truthy), so these conditions are **always true**. This means the conversion always happens regardless, which may mask bugs. Should use `not isinstance(...)`.

Affected locations:
- `_multivariate.py:133,207-213` (in `generate_eps_MV`, `simGBM_MV`)
- `_pa.py:235,301-308,403-410,489-496,962` (in `sd_annualized`, `omega_sharpe_ratio`, `upside_risk`, `downside_deviation`, `_check_ts`)

### 5. Security: API credentials sent over HTTP (`_main_functions.py:883,934`)

```python
url = r"http://api.eia.gov/series/?api_key={}&series_id={}&out=json".format(key, tbl)
url = f"http://api.eia.gov/v2/seriesid/{tbl}?api_key={key}"
```

Both EIA API functions send API keys over unencrypted HTTP. Should use `https://`.

### 6. Security: Bare `except` clauses hide errors (`_main_functions.py:943`, `_sims.py:27`, etc.)

Multiple bare `except:` clauses catch all exceptions silently, hiding real errors:
- `_main_functions.py:943` - EIA v2 API error handling prints but continues
- `_sims.py:27` - `is_iterable()` catches everything
- `_multivariate.py:358-365` - `simOU_MV` silently catches `.to_numpy()` failures
- `data/__init__.py:113-116` - `open_data()` catches load failures silently

---

## Design Issues

### 7. Star imports create namespace pollution (`__init__.py`)

```python
from ._charts import *
from ._pa import *
from ._swap import *
from ._sims import *
from ._main_functions import *
from ._multivariate import *
from .extensions import *
from ._refineryLP import *
```

Every module uses `from .module import *`, exporting all public names to the top-level namespace. This means internal helpers like `Result`, `is_iterable`, `make_into_array`, `shift`, `_MVSIM`, `custom_date_range` are all exported. None of the modules define `__all__`.

### 8. Circular import chain (`_charts.py:1`)

```python
from .__init__ import *  # imports everything from __init__, which imports _charts
```

`_charts.py` imports from `__init__`, which imports `_charts`. This works due to Python's import machinery caching, but is fragile and architecturally incorrect. The chart module should import specific functions it needs from sibling modules directly.

### 9. Module-level data loading in `_swap.py:11`

```python
us_swap = data.open_data("usSwapCurves")
```

This loads a JSON file from disk at **import time**, even if the swap module is never used. This slows down `import risktools` and will cause import failures if the data file is missing.

### 10. `print()` statements in library code (`_sims.py:218,462`)

```python
print("Half-life of theta in days = ", _np.log(2) / theta * bdays_in_year)
```

Both `simOU()` and `simOUJ()` unconditionally print to stdout. Library code should use `logging` or return this information rather than printing.

### 11. Private module naming convention with underscore-prefixed imports

All imports are aliased with underscores (`import pandas as _pd`, `import numpy as _np`) to avoid polluting the namespace from `*` imports. This is a workaround for not defining `__all__`. Defining `__all__` in each module would be cleaner and more Pythonic.

---

## Code Quality Issues

### 12. Deprecated pandas APIs

- `_sims.py:847`: `fillna(method="ffill")` is deprecated in pandas 2.x. Use `ffill()` instead.
- `_swap.py:126`: `.append()` on Index is deprecated. Use `pd.Index([...]).union(dates)` or `pd.concat`.

### 13. Inconsistent comparison patterns

Throughout the codebase, boolean comparisons are done as `if x == True:` or `if isinstance(x, pd.DataFrame) == False:` instead of the idiomatic `if x:` or `if not isinstance(x, pd.DataFrame):`.

### 14. `_check_df()` function does almost nothing (`_main_functions.py:958-963`)

```python
def _check_df(df):
    return df.copy()
```

This function takes a dataframe, copies it, and returns it. The original validation logic is commented out. This adds unnecessary overhead (copying) on every call.

### 15. Duplicate frequency inference logic

The frequency-to-scale mapping logic is duplicated in at least 4 places:
- `_main_functions.py:421-441` (`garch()`)
- `_main_functions.py:966-1009` (`infer_freq()`)
- `_pa.py:238-257` (`sd_annualized()`)
- `_pa.py:943-986` (`_check_ts()`)

This should be consolidated into a single utility function.

### 16. `type` used as parameter name (`_main_functions.py:668`)

```python
def crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call"):
```

`type` shadows the Python built-in. Should be renamed to `option_type` or similar.

### 17. Unused imports

- `_sims.py:4-5`: `ctypes` and `ndpointer` are imported but only used in the dead `_import_csimOU()` function
- `_sims.py:7`: `multiprocessing` is imported but never used
- `_sims.py:8`: `time` is imported but never used
- `_multivariate.py:2`: `from ast import arguments` is imported but never used
- `_charts.py:17`: `arch` is imported as `_arch` (also imported in `_main_functions.py`)

---

## Test Coverage Issues

### 18. Many tests are commented out or `pass`

Several test functions contain only `pass` or have their assertions commented out:
- `test_get_prices()` - entirely `pass`
- `test_swap_irs()` - entirely `pass`
- `test_swap_com()` - entirely `pass`
- `test_chart_spreads()` - entirely `pass`
- `test_stl_decomposition()` - entirely `pass`

This means critical functionality like Morningstar API integration, swap IRS pricing, and STL decomposition have **no active tests**.

### 19. Tests depend on external services

- `test_trade_stats()` calls `yf.download()` to fetch live Yahoo Finance data
- `test_get_eia_df()` calls the live EIA API
- `test_chart_eia_sd()` and `test_chart_eia_steo()` call the live EIA API

These will fail without network access or valid API keys, making CI unreliable.

### 20. No test for performance analytics functions

The `_pa.py` module (987 lines) with functions like `return_cumulative`, `return_annualized`, `sd_annualized`, `omega_sharpe_ratio`, `upside_risk`, `downside_deviation`, `CAPM_beta`, and `timing_ratio` has no dedicated test file. These are only tested indirectly through `test_trade_stats()`.

### 21. No test for charting functions

Most chart functions (`chart_forward_curves`, `chart_five_year_plot`, `chart_perf_summary`, `chart_pairs`, `dist_desc_plot`) have no tests.

### 22. Test path assumptions

`test_sims.py:77` uses `./pytest/data/diffusion.csv` which assumes tests are run from the repository root. This will fail if run from the `pytest/` directory.

---

## Packaging & CI/CD Issues

### 23. `requests` missing from `setup.py` dependencies

The `requirements` list in `setup.py` doesn't include `requests`, but it's used by `_morningstar.py` and `data/__init__.py`. It happens to be installed transitively, but should be explicit.

### 24. Travis CI config is outdated

`.travis.yml` tests Python 3.6, 3.7, 3.8, but `setup.py` declares `python_requires=">=3.7"` and Python 3.6 is EOL. The CI doesn't match the declared compatibility.

### 25. GitHub Actions publish workflow missing test step

The `python-publish.yml` workflow builds and uploads to PyPI but **does not run tests** before publishing. A failing test could result in a broken release.

### 26. No `pytest.ini` or `pyproject.toml` test configuration

There's no pytest configuration file defining test paths, markers, or options. The `pytest/` directory name conflicts with the `pytest` package name.

### 27. `MANIFEST.in` includes unusual file types

```
include *.csv *.docx *.py *.zip *.toml *.c
```

Including `*.docx`, `*.zip`, and `*.c` files in the source distribution is unusual. The generated `sims.c` (1.2MB) is included, which inflates the package.

---

## Documentation Issues

### 28. R-style documentation remnants in docstrings

Several docstrings contain R formatting that doesn't render in Python:
- `_pa.py:162`: `\emph{...}` (R/LaTeX formatting)
- `_pa.py:170-171`: R code examples instead of Python (`data(managers)`, `head(Return.excess(...))`)
- `_pa.py:182,570`: `\eqn{}`, `\deqn{}`, `\code{\link{...}}` (R documentation syntax)

### 29. `__init__.py` is 118 lines of commented-out test code

The module `__init__.py` has ~100 lines of commented-out testing/debugging code that should be removed.

### 30. `stl_decomposition()` docstring is incomplete (`_main_functions.py:752-754`)

```python
"""
Provides a summary of returns distribution using the statsmodels.tsa.seasonal.STL class.
Resamples all data to  prior to STL if
```

The description is cut off mid-sentence.

---

## Performance Considerations

### 31. DataFrame operations in tight loops (`_sims.py:335-349`)

The `_simOUpy()` function uses `DataFrame.iloc` in a tight loop for OU simulation. Each `.iloc` call involves pandas overhead. For the Python fallback path, using raw numpy arrays would be significantly faster.

### 32. Repeated `DataFrame` wrapping in simulation functions

Both `_simOUpy()` and `_simOUJpy()` convert numpy arrays to DataFrames early, then iterate with `.iloc`. The conversion to DataFrame should be deferred until the final return.

---

## Dependency Concerns

### 33. `quandl` package is deprecated

The `quandl` Python package has been deprecated in favor of `nasdaq-data-link`. The `ir_df_us()` function depends on it.

### 34. Heavy dependency chain

The package requires 13+ direct dependencies including `numba`, `arch`, `scikit-learn`, `statsmodels`, `seaborn`, `plotly`, and `matplotlib`. Some of these (like `numba` on Windows) add significant install complexity. Consider making visualization and some analytics dependencies optional.

---

## Summary of Recommendations (Priority Order)

| Priority | Issue | Description |
|----------|-------|-------------|
| **P0** | #1 | Fix `npv()` None check ordering |
| **P0** | #5 | Switch EIA API calls from HTTP to HTTPS |
| **P0** | #3 | Add `raise` to ValueError in `plot_portfolio()` |
| **P1** | #2 | Fix "bear"/"bull" label in `prompt_beta()` chart |
| **P1** | #4 | Replace bitwise `~` with `not` for isinstance checks |
| **P1** | #6 | Replace bare `except` with specific exception types |
| **P1** | #18 | Re-enable or replace commented-out tests |
| **P1** | #25 | Add test step to CI/CD publish workflow |
| **P2** | #7 | Define `__all__` in all modules |
| **P2** | #8 | Fix circular import in `_charts.py` |
| **P2** | #9 | Lazy-load data in `_swap.py` |
| **P2** | #10 | Replace `print()` with `logging` |
| **P2** | #12 | Update deprecated pandas APIs |
| **P2** | #15 | Consolidate frequency inference logic |
| **P2** | #23 | Add `requests` to setup.py dependencies |
| **P3** | #13-17 | Code style cleanup |
| **P3** | #19-22 | Improve test isolation and coverage |
| **P3** | #28-30 | Clean up documentation |
| **P3** | #33 | Migrate from `quandl` to `nasdaq-data-link` |
