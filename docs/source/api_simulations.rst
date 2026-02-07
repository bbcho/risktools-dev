Stochastic Simulation API
=========================

Functions for simulating and fitting stochastic processes commonly used in
commodity and financial price modeling.

See :doc:`guide_stochastic_processes` for the mathematical theory behind these models.

Univariate Simulations
----------------------

.. autofunction:: risktools.simGBM

.. autofunction:: risktools.simOU

.. autofunction:: risktools.simOUJ

Model Fitting
-------------

.. autofunction:: risktools.fitOU

Multivariate Simulations
------------------------

.. autofunction:: risktools.simGBM_MV

.. autofunction:: risktools.simOU_MV

.. autofunction:: risktools.simOUJ_MV

Multivariate Utilities
----------------------

.. autofunction:: risktools.fitOU_MV

.. autofunction:: risktools.generate_eps_MV

.. autofunction:: risktools.calc_spread_MV

Portfolio Analysis
------------------

.. autofunction:: risktools.generate_random_portfolio_weights

.. autofunction:: risktools.calculate_payoffs

.. autofunction:: risktools.simulate_efficient_frontier

.. autofunction:: risktools.make_efficient_frontier_table

.. autofunction:: risktools.plot_efficient_frontier

.. autofunction:: risktools.plot_portfolio
