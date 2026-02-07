Performance Analytics API
=========================

Functions for computing risk-adjusted return metrics, drawdown analysis, and
factor model estimation. These are Python implementations of functions from the
R package `PerformanceAnalytics <https://cran.r-project.org/web/packages/PerformanceAnalytics/>`_
by Peter Carl and Brian G. Peterson.

See :doc:`guide_performance_analytics` for the mathematical theory behind these functions.

Returns
-------

.. autofunction:: risktools.return_cumulative

.. autofunction:: risktools.return_annualized

.. autofunction:: risktools.return_excess

.. autofunction:: risktools.returns

Risk Measures
-------------

.. autofunction:: risktools.sd_annualized

.. autofunction:: risktools.sharpe_ratio_annualized

.. autofunction:: risktools.omega_sharpe_ratio

.. autofunction:: risktools.upside_risk

.. autofunction:: risktools.downside_deviation

Drawdown Analysis
-----------------

.. autofunction:: risktools.drawdowns

.. autofunction:: risktools.find_drawdowns

Factor Models
-------------

.. autofunction:: risktools.CAPM_beta

.. autofunction:: risktools.timing_ratio

Aggregated Statistics
---------------------

.. autofunction:: risktools.trade_stats

Distribution Analysis
---------------------

.. autofunction:: risktools.describe_distribution
