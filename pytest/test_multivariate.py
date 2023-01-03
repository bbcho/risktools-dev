import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

dir = os.path.dirname(os.path.realpath(__file__))

import risktools as rt



def test_calc_spread_MV():

    df = pd.DataFrame(dict(
        A = [1, 2, 3, 4, 5],
        B = [2, 3, 4, 5, 6],
    ))

    ans = rt.calc_spread_MV(df, {'spread': 'A-B'})

    assert ans['spread'].tolist() == [-1, -1, -1, -1, -1], "Spread calculation failed"