Getting Started
===============

Installation
------------

From PyPI
^^^^^^^^^

.. code-block:: bash

   pip install risktools

From Source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/bbcho/risktools-dev.git
   cd risktools-dev
   pip install cython numpy
   pip install -e .

The package includes a Cython extension for high-performance simulation.
Building from source requires a C compiler (gcc on Linux/macOS, MSVC on Windows).

Dependencies
^^^^^^^^^^^^

Core dependencies are installed automatically:

- ``pandas`` -- data manipulation and time series
- ``numpy`` -- numerical computing
- ``scipy`` -- optimization and interpolation
- ``matplotlib``, ``plotly``, ``seaborn`` -- visualization
- ``statsmodels`` -- seasonal decomposition and statistical models
- ``arch`` -- GARCH volatility modeling
- ``scikit-learn`` -- linear regression for CAPM beta
- ``requests`` -- HTTP client for API access

Quick Start
-----------

Import the package:

.. code-block:: python

   import risktools as rt

Loading Bundled Data
^^^^^^^^^^^^^^^^^^^^

risktools ships with over 20 curated datasets. Use :func:`risktools.data.open_data`
to load them:

.. code-block:: python

   # List available datasets
   rt.data.get_names()

   # Load wide-format futures prices
   dfwide = rt.data.open_data('dfwide')
   print(dfwide.head())

   # Load crude oil assay data
   crude = rt.data.open_data('crudeOil')
   print(crude.keys())  # dict with multiple DataFrames

Computing Returns
^^^^^^^^^^^^^^^^^

Convert prices to returns:

.. code-block:: python

   df = rt.data.open_data('dfwide')

   # Relative (percentage) returns
   rets = rt.returns(df[['CL01', 'CL12']], ret_type='rel', period_return=1)

   # Annualized return
   ann_ret = rt.return_annualized(rets['CL01'].dropna(), geometric=True)
   print(f"Annualized return: {ann_ret:.2%}")

   # Annualized volatility
   ann_vol = rt.sd_annualized(rets['CL01'].dropna())
   print(f"Annualized volatility: {ann_vol:.2%}")

Performance Summary Chart
^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize cumulative returns and drawdowns:

.. code-block:: python

   fig = rt.chart_perf_summary(
       rets[['CL01', 'CL12']].dropna(),
       geometric=True,
       title="WTI Front vs 12-Month Spread"
   )
   fig.show()

Simulating Price Paths
^^^^^^^^^^^^^^^^^^^^^^

Generate Monte Carlo paths using Geometric Brownian Motion:

.. code-block:: python

   paths = rt.simGBM(s0=70, mu=0, sigma=0.3, r=0.02, T=1, dt=1/252, sims=100)
   print(f"Simulated {paths.shape[1]} paths over {paths.shape[0]} time steps")

Or a mean-reverting Ornstein-Uhlenbeck process:

.. code-block:: python

   ou_paths = rt.simOU(s0=5, mu=4, theta=2, sigma=0.5, T=2, dt=1/252, sims=50, seed=42)

   # Estimate parameters from a single path
   params = rt.fitOU(ou_paths.iloc[:, 0], dt=1/252, method='OLS')
   print(f"Estimated mu={params['mu']:.2f}, theta={params['theta']:.2f}")

Bond Pricing
^^^^^^^^^^^^

Price a coupon bond:

.. code-block:: python

   # Par bond: coupon rate = YTM
   price = rt.bond(ytm=0.05, c=0.05, T=10, m=2, output='price')
   print(f"Price: ${price:.2f}")  # Should be ~$100

   # Macaulay duration
   dur = rt.bond(ytm=0.05, c=0.05, T=10, m=2, output='duration')
   print(f"Duration: {dur:.2f} years")

Options Pricing
^^^^^^^^^^^^^^^

Price a European call using the CRR binomial model:

.. code-block:: python

   result = rt.crr_euro(s=100, x=105, sigma=0.2, Rf=0.05, T=1, n=100, type='call')
   print(f"Call price: ${result['price']:.2f}")

Refinery Optimization
^^^^^^^^^^^^^^^^^^^^^

Optimize a refinery crude slate:

.. code-block:: python

   data = rt.data.open_data('refineryLPdata')
   result = rt.refineryLP(data['inputs'], data['outputs'])
   print(f"Optimal profit: ${result['profit']:,.0f}")
   print(f"Crude slate: {result['slate']}")

Next Steps
----------

- Browse the :doc:`data_reference` to see all bundled datasets
- Read the conceptual guides for mathematical background:
  - :doc:`guide_performance_analytics`
  - :doc:`guide_stochastic_processes`
  - :doc:`guide_fixed_income`
  - :doc:`guide_options_pricing`
- See the :doc:`api_performance` for complete function signatures
