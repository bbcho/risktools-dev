Charting API
============

Functions for visualizing commodity market data, performance summaries, and
statistical diagnostics.  All chart functions return ``matplotlib`` figure objects
that can be further customized.

Seasonal & Z-Score Analysis
----------------------------

.. autofunction:: risktools.chart_zscore

Five-Year Range
---------------

.. autofunction:: risktools.chart_five_year_plot

Performance Summary
-------------------

.. autofunction:: risktools.chart_perf_summary

Forward Curves
--------------

.. autofunction:: risktools.chart_forward_curves

Pairs & Spreads
---------------

.. autofunction:: risktools.chart_pairs

.. autofunction:: risktools.chart_spreads

EIA Market Data
---------------

.. autofunction:: risktools.chart_eia_sd

.. autofunction:: risktools.chart_eia_steo

Time-Series Decomposition
--------------------------

.. autofunction:: risktools.stl_decomposition

Distribution Diagnostics
-------------------------

.. autofunction:: risktools.dist_desc_plot
