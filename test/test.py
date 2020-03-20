import pyRTL as rtl

key = 'WGDZMootvgNBVyY8hxAy'

# testing

from pandas_datareader import data, wb
from datetime import datetime

spy = data.DataReader("SPY",  "yahoo", datetime(2000,1,1), datetime(2001,6,1))
spy = spy.diff()

out = rtl.trade_stats(df=spy['Adj Close'],Rf=0)
out