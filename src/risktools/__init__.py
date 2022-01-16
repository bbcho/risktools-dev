from ._charts import *
from ._pa import *
from ._swap import *
from ._sims import *
from ._main_functions import *

# from .data import get_gis
from ._refineryLP import *
from ._cullenfrey import describe_distribution

#####################################################################
# TODO
# * Add legend to chart_eia_sd function
#####################################################################


# if __name__ == "__main__":
#     _np.random.seed(42)
#     import json

#     with open("../../user.json") as jfile:
#         userfile = jfile.read()

#     up = json.loads(userfile)

#     username = up["m*"]["user"
#     password = up["m*"]["pass"]

#     print(
#         chart_spreads(
#             [("@HO4H", "@HO4J", "2014"), ("@HO9H", "@HO9J", "2019")],
#             200,
#             username,
#             password,
#             feed="CME_NymexFutures_EOD",
#             output="chart",
#             start_dt="2012-01-01",
#         )
#     )

# sp = simOU()
# print(fitOU(sp))
# print(sp)

# print(bond(output='price'))
# print(bond(output='duration'))
# print(bond(output='df'))

# dflong = data.open_data('dflong')
# ret = returns(df=dflong, ret_type="abs", period_return=1, spread=True).iloc[:,0:2]
# roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade").head(50)

# from pandas_datareader import data, wb
# from datetime import datetime
# df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
# df = df.pct_change()
# df = df.asfreq('B')

# R = df[('Adj Close','SPY')]

# rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
# rt.returns(df = rt.data.open_data('dflong'), ret_type = "log", period_return = 1, spread = True)

# print(y)

# print(_check_ts(R.dropna(), scale=252))

# print(trade_stats(df[('Adj Close','SPY')]))
# print(trade_stats(df['Adj Close']))

# tt = _sr(df['Adj Close'], Rf=0, scale=252)
# print(tt)

# print(drawdowns(df['Adj Close']))
# print(drawdowns(df[('Adj Close','SPY')]))

# rs = find_drawdowns(df['Adj Close'])
# print(rs['SPY']['peaktotrough'])
# print(find_drawdowns(df[('Adj Close','SPY')]))

# print(sharpe_ratio_annualized(df['Adj Close']))
# print(sharpe_ratio_annualized(df[('Adj Close','SPY')]))

# print(omega_sharpe_ratio(df['Adj Close'],MAR=0))
# print(upside_risk(df['Adj Close'], MAR=0))
# print(upside_risk(df[('Adj Close','SPY')], MAR=0))
# print(upside_risk(df['Adj Close'], MAR=0, stat='variance'))
# print(upside_risk(df['Adj Close'], MAR=0, stat='potential'))

# print(downside_deviation(df[('Adj Close','SPY')], MAR=0))
# print(downside_deviation(df['Adj Close'], MAR=0))

# print(return_cumulative(r=df['Adj Close'], geometric=True))
# print(return_cumulative(r=df['Adj Close'], geometric=False))

# print(return_annualized(r=df[('Adj Close','SPY')], geometric=True))
# print(return_annualized(r=df[('Adj Close','SPY')], geometric=False))

# print(return_annualized(r=df['Adj Close'], geometric=True))
# print(return_annualized(r=df['Adj Close'], geometric=False))

# print(sd_annualized(x=df[('Adj Close','SPY')]))
# print(sd_annualized(x=df['Adj Close']))

####################
# R code for testing
# library(timetk)
# library(tidyverse)
# library(PerformanceAnalytics)
# df <- tq_get('SPY', from='2000-01-01', to='2012-01-01')
# df <- df %>% dplyr::mutate(adjusted = adjusted/lag(adjusted)-1)
# R = tk_xts(df, select=adjusted, date_var=date)
# DownsideDeviation(R, MAR=0)
# UpsideRisk(R, 0)
# OmegaSharpeRatio(R, 0)
