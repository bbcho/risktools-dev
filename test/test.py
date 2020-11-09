import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.pandas2ri import rpy2py, py2rpy
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from contextlib import contextmanager

# R vector of strings
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector, Vector

# import R's "base" package
base = importr('base')

utils = importr('utils')
tq = importr('tidyquant')
tv = importr('tidyverse')
ql = importr('Quandl')
rtl = importr('RTL')

def p2r(p_df):
    # Function to convert pandas dataframes to R
    with localconverter(ro.default_converter + pandas2ri.converter) as cv:
        r_from_pd_df = cv.py2rpy(p_df)

    return r_from_pd_df

def r2p(r_df):
    # Function to convert R dataframes to pandas
    with localconverter(ro.default_converter + pandas2ri.converter) as cv:
        pd_from_r_df = cv.rpy2py(r_df)

    return pd_from_r_df
