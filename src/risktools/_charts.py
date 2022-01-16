from .__init__ import *
from . import data
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots as _make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from math import ceil as _ceil
from ._morningstar import *
from statsmodels.tsa.seasonal import STL as _STL
from ._pa import *

from ._main_functions import get_eia_df, infer_freq
from ._cullenfrey import describe_distribution as _desc_dist
import matplotlib.pyplot as _plt
import seaborn as _sns
import arch as _arch
from matplotlib.pyplot import cm as _cm


def chart_zscore(df, freq=None, output="zscore", chart="seasons", **kwargs):
    """
    Function for calculating and plotting the seasonal decomposition and zscore of seasonal data
    Z-score is computed on the residuals conditional on their seasonal period. Beware that most
    seasonal charts in industry (i.e. NG Storage) is not detrended so results once you apply
    STL decomposition will vary from the unadjusted seasonal plot

    Parameters
    ----------
    df : Series
        Series with date and a values column. Datetime index must have a frequency defined (such as 'D', 'B', 'W', 'M' etc...)
    freq : str, optional
        Resampling frequency, by default None. Resamples using mean if downsampling. Not designed for
        upsampling.
    output : ['stl','zscore','seasonal']
        'stl' - for seasonal decomposition object. Run the object method .plot() to plot the STL decomposition.
        Also use the object attributes "observed", "resid", "seasonal", "trend" and "weights" to see components
        and calculate stats.
        'zscore' - return residuals of szore
        'seasonal' - for standard seasonal chart
        By default 'zscore'
    chart : str, optional
        by default 'seasons'

    Returns
    -------
    'stl' - Statsmodels STL object
    'seasonal' - Plotly figure object
    'zscore' - Plotly figure object

    Examples
    --------
    >>> df = rt.data.open_data('eiaStocks')
    >>> df = df.loc[df.series=='NGLower48',['date','value']].set_index('date')['value']
    >>> df = df.resample('W-FRI').mean()
    >>> stl = rt.chart_zscore(df, freq='M')
    >>> stl.plot()
    >>> stl = rt.chart_zscore(df, output='seasonal')
    >>> stl.show()
    >>> stl = rt.chart_zscore(df, output='zscore')
    >>> stl.show()
    """

    # either resample or figure out frequency of the series
    if freq is not None:
        df = df.resample(freq).mean()
    elif df.index.freq is None:
        df.index.freq = pd.infer_freq(df.index[0:10])

    if str(df.index.freqstr)[0] == "M":
        seasonal = 13
    elif str(df.index.freqstr)[0] == "W":
        seasonal = 53
    else:
        seasonal = 7

    stl = _STL(df, seasonal=seasonal, seasonal_deg=0, robust=False).fit()

    if output == "stl":
        return stl
    elif output == "seasonal":
        df = pd.concat([df], keys=["value"], axis=1)
        df["year"] = df.index.year
        df.index = df.index + pd.DateOffset(year=df.year.max())
        df = df.set_index("year", append=True).sort_index()
        df = df.unstack(level=-1).droplevel(0, 1)

        fig = go.Figure()
        for c in df.columns:
            if c == df.index.year.max():
                mode = "lines+markers"
            else:
                mode = "lines"
            fig.add_trace(
                go.Scattergl(
                    x=df[c].dropna().index, y=df[c].dropna(), name=c, mode=mode
                )
            )
        fig.update_layout(xaxis_title="freq", yaxis_title="value")
        fig.update_layout(
            xaxis={
                "tickformat": "%b",
                # tickmode': 'auto',
                "nticks": 13,  # [where value is the max # of ticks]
                # 'tick0': '01', # [where value is the first tick]
                # 'dtick': 1, # [where value is the step between ticks]
            }
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)

        return fig
    elif output == "zscore":
        zs = pd.DataFrame()

        # first calculate mean and std of the residuals by frequency (i.e. by week, month etc...) for all years
        zs["resid"] = stl.resid
        zs["year"] = zs.index.year
        tt = stl.resid.index.to_series()
        zs["per"] = (
            tt.groupby(tt.dt.year).cumcount() + 1
        )  # elegant way to get weeknum, monthnum, quarter etc... regardless of freq
        ag = zs.groupby("per").resid.agg(["mean", "std"]).reset_index()

        zs = (
            zs.reset_index().merge(ag, left_on="per", right_on="per").set_index("date")
        )  # use reset index or it drops the datetime index of zs
        zs["z_score"] = (zs.resid - zs["mean"]) / zs["std"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=zs.index,
                    y=zs.z_score,
                    marker_color=zs.z_score,
                    marker_colorscale=["red", "orange", "green", "orange", "red"],
                    marker_colorbar=dict(title="Z-Score"),
                )
            ]
        )

        fig.update_layout(
            title="Z-Score of Seasonally Adjusted Residuals",
            xaxis_title="date",
            yaxis_title="Z Score",
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)

        return fig


def chart_eia_sd(market, key, start_dt="2010-01-01", output="chart", **kwargs):
    """
    Function for plotting and returning data from the EIA Supply & Demand Balances for
    refined oil products such as mogas, diesel, jet and resid.

    Parameters
    ----------
    market : ['mogas', 'diesel', 'jet', 'resid']
        Refined product type to build the S&D balance for
    key : str
        EIA API key
    start_dt : str | datetime
        Starting date for S&D balance data
    output : ['chart','data']
        Output as a chart or return data as a dataframe, by default 'chart'

    Returns
    -------
    pandas dataframe or a plotly figure object

    Examples
    --------
    >>> import risktools as rt
    >>> fig = rt.chart_eia_sd('mogas', up['eia'])
    >>> fig.show()
    """
    eia = data.open_data("tickers_eia")
    eia = eia[eia.sd_category == market]

    df = get_eia_df(eia.tick_eia.to_list(), key=key)
    df = df.merge(
        eia[["tick_eia", "category"]], left_on=["series_id"], right_on=["tick_eia"]
    ).drop("tick_eia", axis=1)

    df.date = pd.to_datetime(df.date)
    df = df.set_index("date").sort_index()

    # create list of plotly figure objects using repeating calls to the
    # five_year_plot function to loop through later to create subplot
    figs = []
    for c in df.category.unique():
        tf = df.loc[df.category == c, "value"]
        figs.append(chart_five_year_plot(tf, title=c))

    # calc shape of final subplot
    n = len(df.category.unique())
    m = 2
    n = _ceil(n / m)
    fig = _make_subplots(n, m, subplot_titles=(" ", " ", " ", " ", " ", " "),)

    # Copy returns figures from five_year_plot to a single subplot figure
    a = 1
    b = 1
    for i, _ in enumerate(figs):
        for j, _ in enumerate(figs[i]["data"]):
            fig.add_trace(figs[i]["data"][j], row=a, col=b)
            fig["layout"]["annotations"][(a - 1) * 2 + b - 1]["text"] = figs[i][
                "layout"
            ]["title"]["text"]

        # copy xaxis nticks and tickformat to subplots so that it keeps that formatting.
        # if they don't exist, pass
        try:
            fig["layout"][f"xaxis{i+1}"]["nticks"] = figs[i]["layout"]["xaxis"][
                "nticks"
            ]
            fig["layout"][f"xaxis{i+1}"]["tickformat"] = figs[i]["layout"]["xaxis"][
                "tickformat"
            ]
        except:
            pass

        if b == m:
            b = 1
            a += 1
        else:
            b += 1

    fig.update_layout(showlegend=False)
    # return figs
    if output == "chart":
        return fig
    else:
        return df


def chart_five_year_plot(df, **kwargs):
    """
    Function to output a 5 yr range plot for commodity prices

    Parameters
    ----------
    df : Series
        Series or single column dataframe with a datetime index
    """
    df = df.copy()
    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})

    freq = pd.infer_freq(df.index[-10:])

    df["year"] = df.index.year
    start_dt = str(df.index.year.max() - 5) + "-01-01"

    cy = df.index.year.max()
    py = cy - 1

    df = df.sort_index().loc[start_dt:, :]
    fy = df.index.year.min()

    df.index = df.index + pd.DateOffset(year=cy)
    df = df.set_index("year", append=True).sort_index().unstack().droplevel(0, 1)
    df = df.resample(freq).mean()

    # shift back columns where the first row is NaN for offset weeks
    for c in df.columns:
        if np.isnan(df[c].iloc[0]):
            df[c] = df[c].shift(-1)
    df = df.dropna(thresh=2, axis=0)
    mf = df[range(fy, cy)].resample(freq).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mf.min(axis=1).index, y=mf.min(axis=1), line_color="lightgrey", name="min"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mf.max(axis=1).index,
            y=mf.max(axis=1),
            line_color="lightgrey",
            name="max",
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[cy].dropna().index,
            y=df[cy].dropna(),
            name=str(cy),
            line_color="#EF553B",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[py].dropna().index,
            y=df[py].dropna(),
            name=str(py),
            line_color="royalblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mf.mean(axis=1).index,
            y=mf.mean(axis=1),
            name="5 yr mean",
            line_dash="dash",
            line_color="black",
        )
    )

    fig.update_layout(xaxis={"tickformat": "%b", "nticks": 13})
    fig.update_layout(**kwargs)

    return fig


def chart_eia_steo(key, start_dt=None, market="globalOil", output="chart", **kwargs):
    """
    Function to plot the global oil supply and demand balance using the US EIA data

    Parameters
    ----------
    key : str
        Your EIA API key
    start_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    market : ['globalOil']
        What market to build the S&D balance for. For now, only globalOil working.
    output : ['chart', 'data'], optional
        What kind of output to return, by default 'chart'
    **kwargs
        keyword arguments to pass to plotly.graph_objects.figure.update_layout function to modify figure

    Returns
    -------
    output = 'chart'
        Plotly figure object
    output = 'data'
        dataframe

    Examples
    --------
    >>> import risktool as rt
    >>> fig = rt.chart_eia_steo(up['eia'])
    >>> fig.show()
    """
    if market == "globalOil":
        tickers = {
            "STEO.PAPR_NONOPEC.M": "SupplyNOPEC",
            "STEO.PAPR_OPEC.M": "SupplyOPEC",
            "STEO.PATC_WORLD.M": "Demand",
            "STEO.T3_STCHANGE_WORLD.M": "Inv_Change",
        }

    df = get_eia_df(list(tickers.keys()), key=key)
    df["name"] = df["series_id"].map(tickers)
    df = (
        df[["date", "value", "name"]]
        .set_index(["date", "name"])
        .sort_index()
        .unstack()
        .droplevel(0, 1)
        .dropna()
    )
    df["Supply"] = df.SupplyNOPEC + df.SupplyOPEC
    df.Inv_Change *= -1
    df = df.drop(["SupplyNOPEC", "SupplyOPEC"], axis=1)

    if start_dt is not None:
        df = df[start_dt:]

    if output == "data":
        return df
    else:
        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
        fig.update_layout(
            title=f"US EIA {market} Supply and Demand Balance",
            xaxis_title="date",
            yaxis_title="millions of barrels per day",
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)
        return fig


def chart_perf_summary(df, geometric=True, title=None):
    """
    Function to plot the cumulative performance and drawdowns of
    multiple assets

    Parameters
    ----------
    df : DataFrame | Series
        wide dataframe with a datetime index and assets as columns or a
        series with a datetime index for a univariate chart
    geometric : bool
        True to plot geometric returns, False to plot absolute
    title : str
        Title for plot


    >>> import risktools as rt
    >>> df = rt.data.open_data('dfwide')
    >>> df = df[['CL01', 'CL12', 'CL36']]
    >>> df = rt.returns(df, period_return=1)
    >>> rt.chart_perf_summary(df, title="Cummulative Returns and Drawdowns")
    """
    df = df.copy()
    if geometric == True:
        ret = df.add(1).cumprod()
    else:
        ret = df.cumsum()

    dd = drawdowns(df, geometric=geometric)

    cols = DEFAULT_PLOTLY_COLORS

    fig = _make_subplots(
        rows=2, cols=1, subplot_titles=("Cummulative Returns", "Drawdowns")
    )
    for i, c in enumerate(ret.columns):
        fig.add_trace(
            go.Scatter(
                x=ret.index,
                y=ret[c],
                name=c,
                line=dict(width=2, color=cols[i]),
                legendgroup="a",
            ),
            row=1,
            col=1,
        )

    for i, c in enumerate(dd.columns):
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd[c],
                name=c,
                line=dict(width=2, color=cols[i]),
                legendgroup="b",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title_text=title)
    return fig


def chart_forward_curves(
    df,
    code=None,
    cmdty=None,
    resample_freq=None,
    skip=5,
    curve_len=365 * 2,
    yaxis_title="$/BBL",
    **kwargs,
):
    """
    Chart forward curves of a futures contract strip

    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with a datetime index and columns representing the futures contract, ordered by most recent expiry
    code : str, optional
        commodity code to use to filter
    cmdty : str, optional
        expiry_table code to use to get expiry dates to plot forward curves. See data.open_data('expiry_data') for allowable
        values, using the column 'cmdty'
    resample_freq : str, optional
        pandas resample frequency type to downsample by. Assumes that new freq of lower periodicity than old freq. Uses mean to downsample
    skip : int
        number of days between forward curves, must be 1 or greater
    curve_len : int
        how long the draw the curve on the plot, in days. Required if cmdty is not given.
    uom : str
        Unit of Measure to show on the final chart y-axis
    ***kwargs
        keyword arguments to pass to fig.update_layout function for Plotly figures. Mainly
        to adjust figure layout.

    Returns
    -------
    Chart of forward curves varying with time

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.open_data('dfwide')
    >>> rt.chart_forward_curves(df, 'HO', yaxis_title='$/g', skip=10)
    >>> rt.chart_forward_curves_new(df, 'HO', cmdty='cmeulsd', yaxis_title='$/g', skip=2)
    """
    df = df.copy()
    fig = go.Figure()

    # if cmdty is defined, then use the expiry_table table to extract the expiry
    # schedule to pass to the _plot_crv function. Otherwise pass None to the function
    # so that it uses the defined time step
    if isinstance(cmdty, str):
        exp = data.open_data("expiry_table")
        exp = exp.loc[exp.cmdty == cmdty, "Last_Trade"]
    else:
        exp = None

    if code is not None:
        df = df[df.columns[df.columns.str.startswith(code)]]

    if resample_freq is not None:
        df = df.resample(resample_freq).mean()

    def _plot_crv(df, date, d=curve_len, exp=exp):
        """
        Convert series index to align with front contract dates with the curve spanning 400 days thereafter
        df
            dataframe with all contracts to be plotted
        d : int
            number of days that the curve should span

        Returns
        -------
        pandas series to plot with modified datetime index
        """
        df = df.copy()
        df.columns = df.columns.str.replace("[^0-9]", "", regex=False).astype(int)
        se = df.loc[date, :]

        n = len(se)
        days_per_step = _ceil(d / n)

        if exp is None:
            idx = []
            for i, r in se.iteritems():
                idx.append(
                    pd.to_datetime(date) + pd.DateOffset((i - 1) * days_per_step)
                )
        else:
            idx = exp[exp > date]
            idx = idx[: len(se)]

        se.index = pd.Index(idx)

        return se

    cnt = 0
    for i, r in df.iterrows():
        if cnt % skip == 0:
            crv = _plot_crv(df, i)
            fig.add_trace(
                go.Scattergl(
                    x=crv.index,
                    y=crv,
                    line=dict(width=1, dash="dash"),
                    legendgroup="b",
                    showlegend=False,
                    mode="lines",
                )
            )
        cnt += 1

    # add front month trace
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df.iloc[:, 0],
            name="Front Contract",
            legendgroup="a",
            line_color="royalblue",
        )
    )

    # leave this here before **kwargs so that users can adjust
    fig.update_layout(width=800, height=800)

    fig.update_layout(
        dict(title=f"{code} Forward Curves", yaxis_title=yaxis_title, **kwargs,)
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, autorange=True, thickness=0.05), type="date",
        )
    )

    return fig


def chart_pairs(df, title="Time Series Pairs Plot", **kwargs):
    """
    Pairwise scatter matrix plot for timeseries

    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with a datetime index and columns representing the futures contract, ordered by most recent expiry
    title : str, optional
        Chart title, by default "Time Series Pairs Plot"
    **kwargs
        keyword arguments to pass to plotly.graph_objects.Figure.update_layout function
    """
    dt_idx = df.index.name
    df = df.reset_index().copy()

    dims = []
    for c, i in df.iteritems():
        dims.append(dict(label=c, values=df[c],))

    fig = go.Figure()
    fig.add_trace(
        go.Splom(
            dimensions=dims,
            showupperhalf=False,
            marker=dict(color=df[dt_idx].astype(int), colorscale="Portland"),
            diagonal_visible=False,
        )
    )
    fig.update_layout(width=1000, height=1000, title=title)
    if kwargs is not None:
        fig.update_layout(**kwargs)

    return fig


def chart_spreads(
    username,
    password,
    pairs,
    days_from_expiry=200,
    feed=None,
    output="chart",
    start_dt="2012-01-01",
    **kwargs,
):
    """
    Chart commodity price spreads given as a list of tuple with two commodity codes and a name

    Parameters
    ----------
    username : str
        username for Morningstar API
    password : str
        password for Morningstar API
    pairs : List of Tuples with String
        A list of tuples of the format [(code_1, code_2, legend_label), (..,..,..)]. The spread calculated is then
        (code_1 - code_2). For example to plot the March/April time spread for NYMEX ULSD for 2014 & 2019, it would
        be [('@HO4H', '@HO4J', '2014'), ('@HO9H', '@HO9J', '2019')]
    days_from_expiry : int
        Days from expiry of the earliest contract to start the plot from (i.e. 200 days). By default 200
    feed : str
        feed to use to get data from Morningstar
    start_dt : str
        Start date for the data retrival as a string, in the format "yyyy-mm-dd"
    output : ['data', 'chart']
        whether to return data or a chart, by default 'chart'
    **kwargs
        Arguments to pass to the plotly update_layout function to control the layout elements of the final figure. You
        can also modify the returned figure object if required.

    Returns
    -------
    a dataframe or a plotly figure object

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.chart_spreads(
            username=up['m*']['user'],
            password=up['m*']['pass'],
            pairs=[('@HO4H', '@HO4J', '2014'), ('@HO9H', '@HO9J', '2019')],
            feed="CME_NymexFutures_EOD")
    """

    codes = []

    for c in pairs:
        codes.append(c[0])
        codes.append(c[1])

    df = get_prices(
        username=username, password=password, feed=feed, codes=codes, start_dt=start_dt
    )

    df = df.unstack(level=0).settlement_price

    res = pd.DataFrame()
    fig = go.Figure()

    for c in pairs:
        tmp = (df[c[0]] - df[c[1]]).dropna().reset_index(drop=True)[-days_from_expiry:]

        # reset index twice to bring in index number
        tmp = tmp.reset_index().reset_index()
        tmp.columns = ["days_from_exp", "date", "spread"]
        tmp.days_from_exp -= days_from_expiry
        tmp["year"] = c[2]
        res = res.append(tmp)

        if output == "chart":
            fig.add_trace(
                go.Scatter(
                    x=tmp.dropna().days_from_exp, y=tmp.dropna().spread, name=c[2]
                )
            )

    if kwargs is not None:
        fig.update_layout(**kwargs)

    if output == "chart":
        return fig
    else:
        return res


def dist_desc_plot(x, figsize=(10, 10)):
    """
    Provides a summary of returns' distribution

    Parameters
    ----------
    x : Pandas Series
        iterable of numeric returns

    Returns
    -------
    matplotlib Figure object

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.open_data('dflong')
    >>> x = df['BRN01'].pct_change().dropna()
    >>> rt.dist_desc_plot(x)     
    """
    _plt.figure(figsize=figsize)
    ax1 = _plt.subplot2grid((4, 2), (0, 0), rowspan=2)  # return dist
    ax2 = _plt.subplot2grid((4, 2), (2, 0), rowspan=2)  # cullenfrey
    ax3 = _plt.subplot(422)  # daily returns ts
    ax4 = _plt.subplot(424, sharex=ax3)  # garch vol
    ax5 = _plt.subplot(426, sharex=ax3)  # cum ret
    ax6 = _plt.subplot(428, sharex=ax3)  #

    # Returns Histogram
    _sns.histplot(x, ax=ax1, kde=True, stat="density")
    _sns.rugplot(x, ax=ax1)
    norm = np.random.normal(scale=x.std(), size=len(x) * 10)
    _sns.kdeplot(norm, ax=ax1, color="purple")
    ax1.set_xlabel("Returns")
    ax1.set_title("Return Distribution")

    # ax1.axvline(x.std() * -1.65, color="grey", ls="--", lw=1)
    ax1.axvline(np.percentile(x, 5), color="grey", ls="--", lw=1)
    _, ymax = ax1.get_ylim()
    # ax1.text(x.std() * -1.65 * 1.5, 0.8 * ymax, "95% VaR", rotation=90)
    ax1.text(np.percentile(x, 5) * 1.5, 0.8 * ymax, "95% VaR", rotation=90)

    # Cullen and Frey
    _desc_dist(x, boot=500, discrete=False, method="unbiased", ax=ax2)

    color = iter(_cm.rainbow(np.linspace(0, 1, 10)))
    c = next(color)

    # daily returns time series
    ax3.plot(x, c=next(color))
    ax3.set_title("Returns")

    # garch vol
    # *100 to scale it for model based on typical daily returns
    am = _arch.univariate.arch_model(x * 100, vol="Garch",)
    res = am.fit(disp="off").conditional_volatility / 100
    try:
        ann = np.sqrt(infer_freq(x, multiplier=True))
        text = "Annualized "
    except:
        ann = 1
        text = ""
    ax4.fill_between(res.index, res * ann, 0, color=next(color))
    ax4.set_title(f"Garch (1,0,1) {text}Volatility")

    # Cummulative Returns
    ax5.fill_between(x.index, x.cumsum(), 0, color=next(color))
    ax5.set_title("Cummulative Returns")

    # Drawdowns
    dd = drawdowns(x)
    ax6.fill_between(dd.index, dd, 0, color=next(color))
    ax6.set_title("Drawdowns")

    _plt.tight_layout()
