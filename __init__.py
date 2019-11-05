from rpy2.robjects.pandas2ri import ri2py, py2ri

from ._functions.pyRTL_func import ir_df_us
from ._functions.pyRTL_func import simGBM
from ._functions.pyRTL_func import simOU
from ._functions.pyRTL_func import simOUJ
from ._functions.pyRTL_func import fitOU
from ._functions.pyRTL_func import npv_at_risk
from ._functions.pyRTL_func import npv
from ._functions.pyRTL_func import _get_RT_data

_data = _get_RT_data()

cancrudeassays = _data['cancrudeassays']
cancrudeprices = _data['cancrudeprices']
df_fut = _data['df_fut']
dflong = _data['dflong']
dfwide = _data['dfwide']
expiry_table = _data['expiry_table']
holidaysOil = _data['holidaysOil']
tickers_eia = _data['tickers_eia']
_Random_seed = _data['_Random_seed']
