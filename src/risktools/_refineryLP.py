import pandas as _pd
from scipy.optimize import linprog as _linprog


def refineryLP(crude_inputs, product_outputs, return_all=False):
    """
    Refinery optimization LP model
    
    Parameters
    ----------
    crude_inputs : DataFrame
    
    product_outputs : DataFrame

    Returns
    -------

    Examples
    --------
    >>> import risktools as rt
    >>> crudes = rt.data.open_data('ref_opt_inputs')
    >>> products = rt.data.open_data('ref_opt_outputs')
    >>> rt.refineryLP(crudes, products)
    """

    crudes = crude_inputs.copy()
    products = product_outputs.copy()
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

    if return_all == False:
        return dict(profit=-out["fun"], slate=out["x"])
    else:
        return out

