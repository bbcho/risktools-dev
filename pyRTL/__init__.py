from rpy2.robjects.pandas2ri import rpy2py, py2rpy

# from ._functions.pyRTL_func import ir_df_us
# from ._functions.pyRTL_func import simGBM
# from ._functions.pyRTL_func import simOU
# from ._functions.pyRTL_func import simOUJ
# from ._functions.pyRTL_func import fitOU
# from ._functions.pyRTL_func import npv_at_risk
# from ._functions.pyRTL_func import npv
from ._functions.pyRTL_func import _get_RT_data
from ._functions.pyRTL_func import *

_data = _get_RT_data()

cancrudeassays = _data['cancrudeassays']
cancrudeprices = _data['cancrudeprices']
df_fut = _data['df_fut']
dflong = _data['dflong']
dfwide = _data['dfwide']
expiry_table = _data['expiry_table']
holidaysOil = _data['holidaysOil']
tickers_eia = _data['tickers_eia']

ng_storage = _data['ng_storage']
tickers_eia = _data['tickers_eia']
tradeCycle = _data['tradeCycle']
twoott = _data['twoott']
twtrump = _data['twtrump']
usSwapCurves = _data['usSwapCurves']
usSwapCurvesPar = _data['usSwapCurvesPar']
usSwapIR = _data['usSwapIR']
usSwapIRdef = _data['usSwapIRdef']
