import pandas as pd
import os


def ir_df_us(quandlkey=None, ir_sens=0.01):
    """    
    Extracts US Tresury Zero Rates using Quandl
    
    Parameters
    ----------
    quandlkey : {str}
        Your Quandl key "yourkey" as a string
    ir_sens : {float}
        Creates plus and minus IR sensitivity scenarios with specified shock value.

    Returns
    -------
    A pandas data frame of zero rates

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    """
    
    if quandlkey is not None:
        ir = rtl.ir_df_us(quandlkey, ir_sens)
        return(r2p(ir))
    else:
        print('No Quandl key provided')

