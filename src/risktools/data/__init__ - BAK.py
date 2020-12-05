import pandas as pd
import geopandas
import os
import json
import numpy as np

def norm_df(fn):
    rf = pd.DataFrame() # create results df
    
    # read json files with nested dataframes from R
    df = pd.read_json(fn)
    
    for c in df.columns:
        tmp = pd.json_normalize(df[c])

        tmp = pd.concat([tmp.set_index(tmp.columns[0])], keys=[c], axis=1).T
        rf = rf.append(tmp)

    rf.columns.name='specifications'
    rf.index = rf.index.set_names(['crude','cut'])
    return rf.sort_index()

def read_curves(fn):
    # open SwapCurve files (based on R S3 class Discount Curves) and convert to 
    # dictionary with nested arrays, dictionaries and one nested dataframe

    with open(fn) as f:
        dd = json.load(f)

    for key in dd.keys():
        if isinstance(dd[key],list) == True:
            dd[key] = np.array(dd[key])

    dd['table'] = pd.DataFrame(dd['table'])
    dd['table']['date'] = pd.to_datetime(dd['table']['date'], utc=True, unit='D')

    return dd

file_actions = {
    'cancrudeassays.json':{'date_fields':['YM'], 'load_func':pd.read_json},
    'cancrudeassayssum.json':{'date_fields':['YM'], 'load_func':pd.read_json},
    'cancrudeprices.json':{'date_fields':['YM'], 'load_func':pd.read_json},
    'crudeassaysBP.json':{'date_fields':None, 'load_func':norm_df},
    'crudeassaysXOM.json':{'date_fields':None, 'load_func':norm_df},
    'crudepipelines.geojson':{'date_fields':None, 'load_func':geopandas.read_file},
    'crudes.json':{'date_fields':None,'load_func':pd.read_json},
    'dflong.json':{'date_fields':['date'],'load_func':pd.read_json},
    'dfwide.json':{'date_fields':['date'],'load_func':pd.read_json},
    'df_fut.json':{'date_fields':['date'],'load_func':pd.read_json},
    'eiaStocks.json':{'date_fields':['date'],'load_func':pd.read_json},
    'eiaStorageCap.json':{'date_fields':['date'],'load_func':pd.read_json},
    'expiry_table.json':{'date_fields':['Last_Trade','First_Notice','First_Delivery','Last_Delivery'],'load_func':pd.read_json},
    'fizdiffs.json':{'date_fields':['date'],'load_func':pd.read_json},
    'holidaysOil.json':{'date_fields':['value'],'load_func':pd.read_json},
    'planets.json':{'date_fields':None,'load_func':pd.read_json},
    'productspipelines.geojson':{'date_fields':None, 'load_func':geopandas.read_file},
    'productsterminals.geojson':{'date_fields':None, 'load_func':geopandas.read_file},
    'ref.opt.inputs.json':{'date_fields':None,'load_func':pd.read_json},
    'ref.opt.outputs.json':{'date_fields':None,'load_func':pd.read_json},
    'refineries.geojson':{'date_fields':None, 'load_func':geopandas.read_file},
    'tickers_eia.json':{'date_fields':None,'load_func':pd.read_json},
    'tradeCycle.json':{'date_fields':['flowmonth','trade_cycle_end'],'load_func':pd.read_json},
    'tradeprocess.json':{'date_fields':['date'],'load_func':pd.read_json},
    'usSwapCurves.json':{'date_fields':None, 'load_func':read_curves},
    'usSwapCurvesPar.json':{'date_fields':None, 'load_func':read_curves},
    'usSwapIR.json':{'date_fields':['date'],'load_func':pd.read_json},
    'usSwapIRdef.json':{'date_fields':None,'load_func':pd.read_json},
}

path = os.path.dirname(__file__)
files = os.listdir(path)
files = [os.path.join(path, f) for f in files]

for fp in files:
    # assert fn in file_actions.keys(), f'{fn} not defined in file_actions dictionary'
    # get filename to pass to file_actions dict. Note that I use the full file path
    # to open the file
    fn = os.path.basename(fp)
    if fn in file_actions.keys():
        df = file_actions[fn]['load_func'](fp)

        if isinstance(df, pd.DataFrame):
            # convert "." to "_" in column names
            df.columns = df.columns.str.replace('.','_')

        # convert datetime fields
        if file_actions[fn]['date_fields'] is not None:
            for d in file_actions[fn]['date_fields']:
                df[d] = pd.to_datetime(df[d])

        var = fn.replace('.json','').replace('.geojson','').replace('.','_')
        
        # save results to variable named the same as the filename, excluding file extension
        exec(f"{var}=df")
