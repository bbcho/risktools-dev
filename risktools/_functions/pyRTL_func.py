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
    

def npv(init_cost , C, cf_freq, F, T, disc_factors, BreakEven, BE_yield):
    """
    Compute NPV
    
    Parameters
    ----------
    init_cost : Initial investment cost
    C : Periodic cash flow
    cf_freq : Cash flow frequency in year fraction e.g. quarterly = 0.25
    F : Final terminal value
    T : Final maturity in years
    disc_factors : Data frame of discount factors using ir_df_us() function. 
    BreakEven : True when using a flat discount rate assumption.
    BE_yield : Set the flat IR rate when BeakEven = TRUE.

    Returns
    -------    
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float
    
    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(quandlkey = quandlkey,ir.sens=0.01) 
    >>> myDict = rt.npv(init.cost=-375,C=50,cf.freq=.5,F=250,T=2,disc_factors=ir,BreakEven=True,BE_yield=.0399)
    >>> myDict['df']
    >>> myDict['npv']
    """
    tf = rtl.npv(init_cost, C, cf_freq, F, T, disc_factors, BreakEven, BE_yield)
    
    myDict = dict()
    
    myDict['df'] = r2p(tf[0])
    myDict['npv'] = np.array(tf[1])[0]
    
    return(myDict)

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

def swapIRS(trade_date = pd.Timestamp.now(),
            eff_date = pd.Timestamp.now() + pd.DateOffset(days=2),
            mat_date = pd.Timestamp.now() + pd.DateOffset(days=2) + pd.DateOffset(years=2),
            notional = 1000000,
            pay_rec = "Rec",
            fixed_rate=0.05,
            float_curve = usSwapCurves,
            reset_freq = 3,
            disc_curve = usSwapCurves,
            convention = ["act",360],
            bus_calendar="NY",
            output = "price"):
    """
    Commodity swap pricing from exchange settlement
    
    Parameters
    ----------
    trade_date {Timestamp|str}: Defaults to today().
    eff_date {Timestamp|str}: Defaults to today() + 2 days.
    mat_date {Timestamp|str}: Defaults to today() + 2 years.
    notional {long int}: Numeric value of notional. Defaults to 1,000,000.
    pay_rec {str}: "Pay" or "Rec" fixed.
    fixed_rate {float}: fixed interest rate. Defaults to 0.05.
    float_curve {R DicountCurve Obj}: List of interest rate curves. Defaults to data("usSwapCurves").
    reset_freq {int}: 1 = "monthly", 3 = quarterly, 6 = Semi annual 12 = yearly.
    disc_curve {R DicountCurve Obj}: List of interest rate curves. Defaults to data("usSwapCurves").
    convention {list}: Vector of convention e.g. c("act",360) c(30,360),...
    bus_calendar {str}: Banking day calendar. Not implemented.
    output {str}: "price" for swap price or "all" for price, cash flow data frame, duration.
    
    Returns
    -------
    Dictionary with swap price, cash flow data frame and duration.
    
    Examples
    --------
    import risktools as rt
    usSwapCurves = rt.usSwapCurves
    rt.swapIRS(trade_date = "2020-01-04", eff_date = "2020-01-06",mat_date = "2022-01-06", notional = 1000000,pay_rec = "Rec", fixed_rate=0.05, float_curve = usSwapCurves, reset_freq=3,disc_curve = usSwapCurves, convention = ["act",360],bus_calendar = "NY", output = "all")
    """
    
    # convert python dates to R
    if isinstance(trade_date,str) == True:
        tdt = base.as_Date(trade_date)
    else:
        tdt = base.as_Date(trade_date.strftime('%Y-%m-%d'))
        
    if isinstance(eff_date,str) == True:
        edt = base.as_Date(eff_date)
    else:
        edt = base.as_Date(eff_date.strftime('%Y-%m-%d'))
        
    if isinstance(mat_date,str) == True:
        mdt = base.as_Date(mat_date)
    else:
        mdt = base.as_Date(mat_date.strftime('%Y-%m-%d'))
        
            
    x = rtl.swapIRS(
        trade_date = tdt,
        eff_date = edt,
        mat_date = mdt,
        notional = notional,
        PayRec = pay_rec,
        fixed_rate = fixed_rate,
        float_curve = float_curve,
        reset_freq = reset_freq,                
        disc_curve = disc_curve,
        convention = convention,
        bus_calendar = bus_calendar,                
        output = output)
    
    if output == "price":
        out = x[0]
    else:
        out = dict()
        out['price'] = x[0][0]
        out['cashflow'] = r2p(x[1])
        out['duration'] = x[2][0]
        
        out['cashflow']['dates'] = pd.to_datetime(out['cashflow']['dates'], unit='D', utc=True)
        
    return out

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

