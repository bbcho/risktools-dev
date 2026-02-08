import pandas as _pd
from scipy.optimize import linprog as _linprog
from typing import Union

__all__ = [
    "refinery_lp",
    "refineryLP",
]


def refinery_lp(crude_inputs: _pd.DataFrame, product_outputs: _pd.DataFrame, return_all: bool = False) -> Union[dict, 'scipy.optimize.OptimizeResult']:
    """
    Refinery optimization LP model

    Parameters
    ----------
    crude_inputs : DataFrame

    product_outputs : DataFrame

    return_all : bool
        If False (default), return a dict with profit and slate.
        If True, return the full scipy OptimizeResult.

    Returns
    -------
    dict or scipy.optimize.OptimizeResult

    Examples
    --------
    >>> import risktools as rt
    >>> crudes = rt.data.open_data('ref_opt_inputs')
    >>> products = rt.data.open_data('ref_opt_outputs')
    >>> rt.refinery_lp(crudes, products)
    """

    crudes = crude_inputs.copy()
    products = product_outputs.copy()
    crudes["info"] = crudes["info"].str.replace(".", "_", regex=False)
    crudes = crudes.set_index("info")

    gpw = _pd.DataFrame(
        dict(
            element=["gross_product_worth", "crude_cost", "processing"],
            light_sweet=[
                (products.prices * products.LightSweet_yield).sum(),
                crudes.loc["price", "LightSweet"],
                crudes.loc["processing_fee", "LightSweet"],
            ],
            heavy_sour=[
                (products.prices * products.HeavySour_yield).sum(),
                crudes.loc["price", "HeavySour"],
                crudes.loc["processing_fee", "HeavySour"],
            ],
        )
    )

    gpw = gpw[["light_sweet", "heavy_sour"]].sum()
    constraints = products[
        ["product", "LightSweet_yield", "HeavySour_yield", "max_prod"]
    ]

    out = _linprog(-gpw, A_ub=constraints.iloc[:, [1, 2]], b_ub=constraints.iloc[:, 3])

    if not return_all:
        return dict(profit=-out["fun"], slate=out["x"])
    else:
        return out


def refineryLP(*args, **kwargs):
    """Deprecated: Use refinery_lp() instead."""
    import warnings
    warnings.warn("refineryLP is deprecated, use refinery_lp instead. Will be removed in v3.0.", DeprecationWarning, stacklevel=2)
    return refinery_lp(*args, **kwargs)
