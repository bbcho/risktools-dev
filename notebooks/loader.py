import json, os
import pandas as pd

with open("../../user.json") as js:
    up = json.load(js)


def _load_json(fn, dataframe=True):
    path = "../pytest"
    fp = os.path.join(path, fn)
    with open(fp) as js:
        if dataframe == True:
            df = pd.read_json(js)
            print(df)
            df.columns = df.columns.str.replace(".", "_", regex=False)
        else:
            df = json.load(js)

    return df
