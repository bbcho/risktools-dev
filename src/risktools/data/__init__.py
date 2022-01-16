import pandas as _pd
import os as _os
import json as _json
import numpy as _np
import requests as _requests
from io import BytesIO as _BytesIO
from zipfile import ZipFile as _ZipFile
import warnings as _warnings


def get_gis(url="https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip"):
    """
    Returns a SpatialPointsDataFrame from a shapefile URL. Examples with EIA and Government of Alberta

    US Energy Information Agency:
    
        EIA crude pipelines : https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip

        EIA Refinery Map : https://www.eia.gov/maps/map_data/Petroleum_Refineries_US_EIA.zip
        
        EIA Products Pipelines : https://www.eia.gov/maps/map_data/PetroleumProduct_Pipelines_US_EIA.zip
        
        EIA Products Terminals : https://www.eia.gov/maps/map_data/PetroleumProduct_Terminals_US_EIA.zip
        
        EIA NG Pipelines : https://www.eia.gov/maps/map_data/NaturalGas_InterIntrastate_Pipelines_US_EIA.zip
        
        EIA NG Storage : https://www.eia.gov/maps/map_data/PetroleumProduct_Terminals_US_EIA.zip
        
        EIA NG Hubs : https://www.eia.gov/maps/map_data/NaturalGas_TradingHubs_US_EIA.zip
        
        EIA LNG Terminals : https://www.eia.gov/maps/map_data/Lng_ImportExportTerminals_US_EIA.zip

    Alberta Oil Sands, Petroleum and Natural Gas

        AB : https://gis.energy.gov.ab.ca/GeoviewData/OS_Agreements_Shape.zip

    Parameters
    ----------
    url : str
        URL of the zipped shapefile

    Return
    ------
    Returns geopandas object

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.get_gis("https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip")
    """

    try:
        import geopandas as _geopandas
    except:
        raise ImportError("Geopandas not installed. Please install before running")

    try:
        from fiona.io import ZipMemoryFile as _ZMF
    except:
        raise ImportError("Fiona not installed. Please install before running")

    fn = _requests.get(url)

    # useful for when there are multiple directories or files. Takes first shape file
    for ff in _ZipFile(_BytesIO(fn.content)).namelist():
        if ff[-4:] == ".shp":
            shp_file = ff
            break

    zf = _ZMF(fn.content)
    shp = zf.open(shp_file)

    return _geopandas.GeoDataFrame.from_features(shp, crs=shp.crs)


def get_names():
    """
    return valid names for the open_data() function.

    Returns
    -------
    List of strings
    
    Examples
    --------
    >>> import risktools as rt
    >>> rt.get_names()
    """
    return list(_file_actions.keys())


def open_data(nm):
    """
    Function used to return built-in datasets from risktools. To get a list of valid datasets, use the get_names() function.

    Parameters
    ----------
    nm : str
        Name of dataset to return
    
    Returns
    -------
    Varies by data requests

    Examples
    --------
    >>> import risktools as rt
    >>> rt.open_data('cancrudeassays')
    
    """
    fn = ""
    path = _os.path.dirname(__file__)
    try:
        fn = _file_actions[nm]["file"]
    except ValueError:
        print(f"{nm} is not a valid file name to open")

    fp = _os.path.join(_path, fn)
    try:
        df = _file_actions[nm]["load_func"](fp)
    except:
        _warnings.warn(f"File actions for {nm} not defined. Running default behavior.")
        df = _pd.read_json(_os.path.join(_path, f"{nm}.json"))

    if isinstance(df, _pd.DataFrame):
        # convert "." to "_" in column names
        df.columns = df.columns.str.replace(".", "_", regex=False)

    # convert datetime fields
    if _file_actions[nm]["date_fields"] is not None:
        for d in _file_actions[nm]["date_fields"]:
            df[d] = _pd.to_datetime(df[d])

    if "index" in _file_actions[nm].keys():
        df = df.set_index(_file_actions[nm]["index"]).sort_index()
        if len(df.columns) < 2:
            df = df.iloc[:, 0]

    return df


def _norm_df(fn):
    rf = _pd.DataFrame()  # create results df

    # read json files with nested dataframes from R
    df = _pd.read_json(fn)

    for c in df.columns:
        tmp = _pd.json_normalize(df[c])

        tmp = _pd.concat([tmp.set_index(tmp.columns[0])], keys=[c], axis=1).T
        rf = rf.append(tmp)

    rf.columns.name = "specifications"
    rf.index = rf.index.set_names(["crude", "cut"])
    return rf.sort_index()


def _read_curves(fn):
    # open SwapCurve files (based on R S3 class Discount Curves) and convert to
    # dictionary with nested arrays, dictionaries and one nested dataframe

    with open(fn) as f:
        dd = _json.load(f)

    for key in dd.keys():
        if isinstance(dd[key], list) == True:
            dd[key] = _np.array(dd[key])

    dd["table"] = _pd.DataFrame(dd["table"])
    dd["table"]["date"] = _pd.to_datetime(dd["table"]["date"], utc=True, unit="D")

    return dd


_file_actions = {
    "cancrudeassays": {
        "file": "cancrudeassays.json",
        "date_fields": ["YM"],
        "load_func": _pd.read_json,
    },
    "cancrudeassayssum": {
        "file": "cancrudeassayssum.json",
        "date_fields": ["YM"],
        "load_func": _pd.read_json,
    },
    "cancrudeprices": {
        "file": "cancrudeprices.json",
        "date_fields": ["YM"],
        "load_func": _pd.read_json,
    },
    "crudeassaysBP": {
        "file": "crudeassaysBP.json",
        "date_fields": None,
        "load_func": _norm_df,
    },
    "crudeassaysXOM": {
        "file": "crudeassaysXOM.json",
        "date_fields": None,
        "load_func": _norm_df,
    },
    # "crudepipelines": {
    #     "file": "crudepipelines.geojson",
    #     "date_fields": None,
    #     "load_func": _geopandas.read_file,
    # },
    "crudes": {"file": "crudes.json", "date_fields": None, "load_func": _pd.read_json},
    "dflong": {
        "file": "dflong.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
        "index": ["series", "date"],
    },
    "dfwide": {
        "file": "dfwide.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
        "index": "date",
    },
    "df_fut": {
        "file": "df_fut.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
        "index": ["series", "date"],
    },
    "eiaStocks": {
        "file": "eiaStocks.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "eiaStorageCap": {
        "file": "eiaStorageCap.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "expiry_table": {
        "file": "expiry_table.json",
        "date_fields": [
            "Last_Trade",
            "First_Notice",
            "First_Delivery",
            "Last_Delivery",
        ],
        "load_func": _pd.read_json,
    },
    "fizdiffs": {
        "file": "fizdiffs.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "holidaysOil": {
        "file": "holidaysOil.json",
        "date_fields": ["value"],
        "load_func": _pd.read_json,
    },
    "planets": {
        "file": "planets.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    # "productspipelines": {
    #     "file": "productspipelines.geojson",
    #     "date_fields": None,
    #     "load_func": _geopandas.read_file,
    # },
    # "productsterminals": {
    #     "file": "productsterminals.geojson",
    #     "date_fields": None,
    #     "load_func": _geopandas.read_file,
    # },
    "ref_opt_inputs": {
        "file": "ref.opt.i_nputs.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "ref_opt_outputs": {
        "file": "ref.opt.outputs.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    # "refineries": {
    #     "file": "refineries.geojson",
    #     "date_fields": None,
    #     "load_func": _geopandas.read_file,
    # },
    "tickers_eia": {
        "file": "tickers_eia.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "tradeCycle": {
        "file": "tradeCycle.json",
        "date_fields": ["flowmonth", "trade_cycle_end"],
        "load_func": _pd.read_json,
    },
    "tradeprocess": {
        "file": "tradeprocess.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "usSwapCurves": {
        "file": "usSwapCurves.json",
        "date_fields": None,
        "load_func": _read_curves,
    },
    "usSwapCurvesPar": {
        "file": "usSwapCurvesPar.json",
        "date_fields": None,
        "load_func": _read_curves,
    },
    "usSwapIR": {
        "file": "usSwapIR.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
        "index": "date",
    },
    "usSwapIRdef": {
        "file": "usSwapIRdef.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "ref_opt_inputs": {
        "file": "ref_opt_inputs.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "ref_opt_outputs": {
        "file": "ref_opt_outputs.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
}

_path = _os.path.dirname(__file__)

