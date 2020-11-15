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
    
    
def chart_perf_summary(df, geometric=True, main="Cumulative Returns and Drawdowns", linesize=1.25):
    """
    Multi Asset Display of Cumulative Performance and Drawdowns

    Parameters
    ----------
    df{DataFrame}: Wide dataframe univariate or multivariate of percentage returns
    geometric{bool}: Use geometric returns True or False
    main{str}: Chart title    
    linesize{float}: Size of lines in chart and legend

    Returns
    -------
    Cumulative performance and drawdown charts as png binary

    Examples
    --------
    import risktools as rt
    df = rt.dflong[rt.dflong.series.isin(["CL01","CL12","CL36"])]
    df = rt.returns(df,retType="rel",period_return=1,spread=True)
    df = rt.rolladjust(df,commodityname=["cmewti"],rolltype=["Last.Trade"])
    rt.chart_perf_summary(df, geometric=True, main="Cumulative Returns and Drawdowns",linesize=1.25)
    """
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
        
    x = r2p(rtl.chart_PerfSummary(p2r(df), geometric, main, linesize))
    return image_png(x)


##################
# r_inline_plot needed to display outputs from graphics library in R
# for chart_fwd_curves
##################

@contextmanager
def r_inline_plot(width=600, height=600, dpi=100):

    with grdevices.render_to_bytesio(grdevices.png, 
                                     width=width,
                                     height=height, 
                                     res=dpi) as b:

        yield

    data = b.getvalue()
    display(Image(data=data, format='png', embed=True))

def chart_fwd_curves(df, cmdty = 'cmewti', weekly=False, width = 1024, height = 896, dpi = 150, **kwargs):
    """    
    Returns a plot of forward curves through time
    
    Parameters
    ----------
    df{DataFrame} -- Wide dataframe with date column and multiple series columns (multivariate)
    cmdty{str} -- Futures contract code in expiry_table object: unique(expiry_table$cmdty)
    weekly{bool} -- True if you want weekly forward curves
    width {int} -- width in pixels (needed to display in IPython)
    height {int} -- height in pixels (needed to display in IPython)
    dpi {int} -- dpi in pixels (needed to display in IPython)
    *kwargs -- other NAMED graphical parameters to pass to R graphics package
    
    Returns
    -------
    plot of forward curves through time
    
    Examples
    --------
    import risktools as rt
    rt.chart_fwd_curves(df=rt.dfwide,cmdty="cmewti",weekly=True, main="WTI Forward Curves",ylab="$ per bbl",xlab="",cex=2)
    """
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    with r_inline_plot(width=width, height=height, dpi=dpi):
        r2p(rtl.chart_fwd_curves(p2r(df), cmdty, weekly, **kwargs))
    
def dist_desc_plot(df, width = 1024, height = 896, dpi = 150):
    """
    Provides a summary of returns distribution
    
    Parameters
    ----------
    df {DataFrame} -- Wide dataframe with date column/index and single series (univariate)
    width {int} -- width in pixels (needed to display in IPython)
    height {int} -- height in pixels (needed to display in IPython)
    dpi {int} -- dpi in pixels (needed to display in IPython)
    
    Returns
    -------
    Multiple plots describing the distribution
    
    Examples
    --------
    import risktools as rt
    df = rt.dflong[rt.dflong.series == "CL01"]
    df = rt.returns(df,retType="rel",period_return=1,spread=True)
    df = rt.rolladjust(df,commodityname=["cmewti"],rolltype=["Last.Trade"])
    rt.dist_desc_plot(df)
    """
    
    if isinstance(df.index,pd.DatetimeIndex):
        df = df.reset_index()
    
    with r_inline_plot(width=width, height=height, dpi=dpi):
        rtl.distdescplot(p2r(df))

# Not ready yet, eiaStocks not in main CRAN pkg 
# def chart_pairs(df, title = 'TIme Series Pairs Plot'):
#     """
#     Pairwise scatter chart for timeseries
    
#     Parameters
#     ----------
    
#     df {DataFrame} -- Wide data frame
#     title {str} -- Chart titles
    
#     Returns
#     -------
    
#     plotly figure object
    
#     Examples

#     """



# Not ready yet, eiaStocks not in main CRAN pkg    
# def chart_zscore(df, title = "NG Storage Z Score", per = "yearweek", output = "zscore", chart = "seasons"):
#     """
#     Supports analytics and display of seasonal data. Z-Score is
#     computed on residuals conditional on their seasonal period.
#     Beware that most seasonal charts in industry e.g. (NG Storage)
#     is not detrended so results once you apply an STL decompostion
#     will vary from the unajusted seasonal plot.

#     Parameters
#     ----------
    
#     df {DataFrame} -- Long data frame with columns series, date and value
#     title {str} -- Default is a blank space returning the unique value in df$series.
#     per {str} -- Frequency of seasonality "yearweek" (DEFAULT). "yearmonth", "yearquarter"
#     output {str} -- "stl" for STL decomposition chart, "stats" for STL statistical test results. "zscore" for residuals Z-score, "seasonal" for standard seasonal chart.
#     chart {str} -- "seasons" for feasts::gg_season() (DEFAULT). "series" for feasts::gg_subseries()

#     Returns
#     -------
    
#     Time series of STL decomposition residuals Z-Scores, or standard seasonal chart with feast package.

#     Examples
#     --------
    
#     import risktools as rt
#     df = rt.eiaStocks[rt.eiaStocks.series == "NGLower48"]
#     rt.chart_zscore(df = df, title = "NG Storage Z Score", per = "yearweek", output = "stl", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score", per = "yearweek", output = "stats", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score" ,per = "yearweek", output = "zscore", chart = "seasons")
#     rt.chart_zscore(df = df, title = "NG Storage Z Score" ,per = "yearweek", output = "seasonal", chart = "seasons")
#     """

#     if isinstance(df.index,pd.DatetimeIndex):
#         df = df.reset_index()

#     x = rtl.chart_zscore(p2r(df), title, per, output, chart)
    
#     if output == "stl":
#         return image_png(x)

# def chart_pairs(df, title):
#     """
#     import risktools as rt
#     df = rt.dfwide.reset_index()[['date','CL01','NG01','HO01','RB01']]
#     """
#     x = rtl.chart_pairs(p2r(df), title)

