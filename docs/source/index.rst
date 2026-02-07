risktools documentation
=======================

**risktools** is a Python library for commodity trading analytics and financial risk
management. It is a port of the R package
`RTL <https://cran.r-project.org/web/packages/RTL/index.html>`_ (Risk Tools Library),
designed to support the delivery of finance courses at the Alberta School of Business.

The library provides tools for:

- **Performance Analytics** -- Annualized returns, Sharpe ratios, drawdown analysis,
  CAPM beta estimation, and other risk-adjusted performance metrics.
- **Stochastic Simulation** -- Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck (OU),
  and OU with jump-diffusion (OUJ) processes for price modeling.
- **Fixed Income** -- Bond pricing, duration, net present value, and interest rate swap
  valuation.
- **Options Pricing** -- Cox-Ross-Rubinstein (CRR) binomial tree model for European options.
- **Charting** -- Seasonal decomposition, z-score analysis, forward curves, five-year
  range plots, and distribution diagnostics.
- **Refinery Optimization** -- Linear programming for crude slate optimization.
- **Bundled Data** -- 20+ curated datasets of futures prices, expiry schedules, swap
  curves, crude assays, and EIA supply/demand data.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   data_reference

.. toctree::
   :maxdepth: 2
   :caption: Conceptual Guides

   guide_performance_analytics
   guide_stochastic_processes
   guide_fixed_income
   guide_options_pricing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_performance
   api_simulations
   api_fixed_income
   api_charting
   api_refinery
   api_data

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
