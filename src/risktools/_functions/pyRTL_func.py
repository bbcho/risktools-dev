string = """
npv.at.risk <- function(init.cost, C.cost, cf.freq, F, T, disc.factors, simC, X) {

    df <- tibble(t = seq(from = 0, to = T, by = cf.freq), cf = simC) %>% 
      dplyr::mutate(cf = case_when(cf >= X  ~ (cf - C.cost),cf < X ~ 0),
                    cf = replace(cf, t == 0, init.cost), cf = replace(cf, t == T, last(simC)+F),
                    df = spline(x = disc.factors$maturity, y = disc.factors$discountfactor, xout = t)$y, 
                    pv = cf * df)
    
  x = list(df = df, npv = sum(df$pv))
  return(x)
}
"""
    

def npv_at_risk(init_cost , C_cost, cf_freq, F, T, disc_factors, simC, X):
    """
    NPV at risk function. Replacement for NPV function that takes a list of periodic cashflows as an option instead of a single value
    
    Parameters
    ----------
    init_cost : Initial cost of project (typically negative)
    C_cost : period cost or OPEX
    cf_freq : cash flow frequency in fractions of years (or unit of measure for time to maturity T)
    F : terminal value at maturity
    T : time to maturity (typically years)
    disc_factors : discount factors from interest rate curve
    simC : list of cash flows. Must equal number of periods and formated as Python list
    X : Lower cut off threshold
    
    Returns
    -------
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float
    
    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = rt.npv_at_risk(init.cost=-375,C_cost=5,cf.freq=.5,F=250,T=1,disc_factors=ir,simC=[50,50,50], X=5)
    >>> myDict['df']
    >>> myDict['npv']
    """
    simC = np.asarray(simC)
    simC = ro.FloatVector(simC)
    #disc_factors = ro.FloatVector(disc_factors)
    tf = _npv_at_risk.npv_at_risk(init_cost, C_cost, cf_freq, F, T, disc_factors, simC, X)
    
    myDict = dict()
    
    myDict['df'] = r2p(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)
    
    

