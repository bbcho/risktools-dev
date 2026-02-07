Fixed Income & Derivatives API
==============================

Functions for bond pricing, net present value, options pricing, interest rate
swap valuation, and commodity swap analysis.

See :doc:`guide_fixed_income` for the theory behind bond pricing, duration, and
net present value.  See :doc:`guide_options_pricing` for the CRR binomial model.

Bond Pricing
------------

.. autofunction:: risktools.bond

Net Present Value
-----------------

.. autofunction:: risktools.npv

Options Pricing
---------------

.. autofunction:: risktools.crr_euro

Interest Rate Swaps
-------------------

.. autofunction:: risktools.swap_irs

.. autofunction:: risktools.get_ir_swap_curve

Commodity Swaps
---------------

.. autofunction:: risktools.swap_com

.. autofunction:: risktools.swap_info

.. autofunction:: risktools.swap_fut_weight

Utilities
---------

.. autofunction:: risktools.ir_df_us

.. autofunction:: risktools.roll_adjust

.. autofunction:: risktools.garch

.. autofunction:: risktools.prompt_beta

.. autofunction:: risktools.infer_freq
