# risktools - Python port of R's RTL (Risk Tools Library)
# Commodity trading analytics and financial risk management

from . import data

# --- Performance Analytics ---
from ._pa import (
    return_cumulative,
    return_annualized,
    return_excess,
    sd_annualized,
    omega_sharpe_ratio,
    upside_risk,
    downside_deviation,
    sharpe_ratio_annualized,
    drawdowns,
    find_drawdowns,
    capm_beta,
    timing_ratio,
)

# --- Main Functions ---
from ._main_functions import (
    ir_df_us,
    bond,
    trade_stats,
    returns,
    roll_adjust,
    garch,
    prompt_beta,
    npv,
    crr_euro,
    stl_decomposition,
    get_eia_df,
    infer_freq,
)

# --- Simulations ---
from ._sims import (
    sim_gbm,
    sim_ou,
    sim_ouj,
    fit_ou,
)

# --- Multivariate Simulations ---
from ._multivariate import (
    calc_spread_mv,
    fit_ou_mv,
    generate_eps_mv,
    sim_gbm_mv,
    sim_ou_mv,
    sim_ouj_mv,
    generate_random_portfolio_weights,
    calculate_payoffs,
    simulate_efficient_frontier,
    make_efficient_frontier_table,
    plot_efficient_frontier,
    plot_portfolio,
    MvGbm,
    MvOu,
)

# --- Cython Extensions ---
from .extensions import *

# --- Swap Pricing ---
from ._swap import (
    swap_irs,
    swap_com,
    swap_info,
    swap_fut_weight,
    get_ir_swap_curve,
)

# --- Charting ---
from ._charts import (
    chart_zscore,
    chart_eia_sd,
    chart_five_year_plot,
    chart_eia_steo,
    chart_perf_summary,
    chart_forward_curves,
    chart_pairs,
    chart_spreads,
    dist_desc_plot,
)

# --- Refinery Optimization ---
from ._refineryLP import refinery_lp

# --- Distribution Analysis ---
from ._cullenfrey import describe_distribution

# --- Morningstar API ---
from ._morningstar import get_prices, get_curves

# --- Deprecated Aliases (will be removed in v3.0) ---
# These provide backwards compatibility for code using the old camelCase names.
from ._pa import CAPM_beta
from ._sims import simGBM, simOU, simOUJ, fitOU
from ._multivariate import (
    calc_spread_MV,
    fitOU_MV,
    generate_eps_MV,
    simGBM_MV,
    simOU_MV,
    simOUJ_MV,
    MVGBM,
    MVOU,
)
from ._refineryLP import refineryLP

__all__ = [
    # Data
    "data",
    # Performance Analytics
    "return_cumulative",
    "return_annualized",
    "return_excess",
    "sd_annualized",
    "omega_sharpe_ratio",
    "upside_risk",
    "downside_deviation",
    "sharpe_ratio_annualized",
    "drawdowns",
    "find_drawdowns",
    "capm_beta",
    "timing_ratio",
    # Main Functions
    "ir_df_us",
    "bond",
    "trade_stats",
    "returns",
    "roll_adjust",
    "garch",
    "prompt_beta",
    "npv",
    "crr_euro",
    "stl_decomposition",
    "get_eia_df",
    "infer_freq",
    # Simulations
    "sim_gbm",
    "sim_ou",
    "sim_ouj",
    "fit_ou",
    # Multivariate Simulations
    "calc_spread_mv",
    "fit_ou_mv",
    "generate_eps_mv",
    "sim_gbm_mv",
    "sim_ou_mv",
    "sim_ouj_mv",
    "generate_random_portfolio_weights",
    "calculate_payoffs",
    "simulate_efficient_frontier",
    "make_efficient_frontier_table",
    "plot_efficient_frontier",
    "plot_portfolio",
    "MvGbm",
    "MvOu",
    # Swap Pricing
    "swap_irs",
    "swap_com",
    "swap_info",
    "swap_fut_weight",
    "get_ir_swap_curve",
    # Charting
    "chart_zscore",
    "chart_eia_sd",
    "chart_five_year_plot",
    "chart_eia_steo",
    "chart_perf_summary",
    "chart_forward_curves",
    "chart_pairs",
    "chart_spreads",
    "dist_desc_plot",
    # Refinery Optimization
    "refinery_lp",
    # Distribution Analysis
    "describe_distribution",
    # Morningstar API
    "get_prices",
    "get_curves",
    # Deprecated Aliases
    "CAPM_beta",
    "simGBM",
    "simOU",
    "simOUJ",
    "fitOU",
    "calc_spread_MV",
    "fitOU_MV",
    "generate_eps_MV",
    "simGBM_MV",
    "simOU_MV",
    "simOUJ_MV",
    "MVGBM",
    "MVOU",
    "refineryLP",
]
