import pandas as _pd
import os as _os
import json as _json
import numpy as _np
import requests as _requests
from io import BytesIO as _BytesIO
from zipfile import ZipFile as _ZipFile
import warnings as _warnings
from io import BytesIO
from pandas.errors import ParserError


def get_gis(url="https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip"):
    """
    Returns a SpatialPointsDataFrame from a shapefile URL. Examples with EIA and Government of Alberta

    US Energy Information Agency:
    - EIA crude pipelines : https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip
    - EIA Refinery Map : https://www.eia.gov/maps/map_data/Petroleum_Refineries_US_EIA.zip
    - EIA Products Pipelines : https://www.eia.gov/maps/map_data/PetroleumProduct_Pipelines_US_EIA.zip
    - EIA Products Terminals : https://www.eia.gov/maps/map_data/PetroleumProduct_Terminals_US_EIA.zip
    - EIA NG Pipelines : https://www.eia.gov/maps/map_data/NaturalGas_InterIntrastate_Pipelines_US_EIA.zip
    - EIA NG Storage : https://www.eia.gov/maps/map_data/PetroleumProduct_Terminals_US_EIA.zip
    - EIA NG Hubs : https://www.eia.gov/maps/map_data/NaturalGas_TradingHubs_US_EIA.zip
    - EIA LNG Terminals : https://www.eia.gov/maps/map_data/Lng_ImportExportTerminals_US_EIA.zip

    Alberta Oil Sands, Petroleum and Natural Gas
    - AB : https://gis.energy.gov.ab.ca/GeoviewData/OS_Agreements_Shape.zip

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
    >>> rt.data.get_names()
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
    >>> rt.data.open_data('crudeOil')
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
        df = _load_data(_os.path.join(_path, f"{nm}.json"))

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

    with open(fn, mode="r") as file:
        tmp = _json.load(file)

    df = dict()

    for key in tmp.keys():
        
        # if structure of dict is {key : DataFrame}
        try: 
            df[key] = _pd.DataFrame.from_records(tmp[key])
            # df[key] = _try_dates(df[key])
        except:
            pass

        # if structure of dict is {first_key : {second_key : DataFrame}}
        # return {first_key : DataFrame} where second_key is a column
        # of the DataFrame
        try:    
            cf = _pd.DataFrame()
            for sec_key in tmp[key].keys():
                tf = _pd.DataFrame.from_records(tmp[key][sec_key])
                tf.columns = tf.columns.str.replace("\.+", "_", regex=True)
                cols = list(tf.columns)
                tf['assay'] = sec_key
                tf = tf[['assay', *cols]]
                cf = _pd.concat([cf, tf], axis=0)
            df[key] = cf
            # df[key] = _try_dates(df[key])

        except:
            pass
    
    return df


def _load_data(fn):

    with open(fn) as f:
        dd = _json.load(f)

    try:
        df = _pd.DataFrame.from_records(dd)
        df = _try_dates(df)
        df.columns = df.columns.str.replace("\.+", "_", regex=True)
        return df
    except:
        pass

    for key in dd.keys():

        try:
            dd[key] = _pd.DataFrame.from_records(dd[key])
            dd[key].columns = dd[key].columns.str.replace("\.+", "_", regex=True)
        except:
            pass

        try:
            dd[key] = _pd.DataFrame(dd[key])
            dd[key].columns = dd[key].columns.str.replace("\.+", "_", regex=True)
        except:
            pass

        if isinstance(dd[key], _pd.DataFrame) == True:
            dd[key] = _try_dates(dd[key])

        # try:
        #     dd[key] = _np.array(dd[key])
        # except:
        #     pass
    # dd.columns = dd.columns.str.replace("\.+", "_", regex=True)
    return dd


def _try_dates(df):
    for c in df.columns[df.dtypes == "object"]:  # don't cnvt num
        try:
            df[c] = _pd.to_datetime(df[c])
        except (ParserError, ValueError):  # Can't cnvrt some
            pass  # ...so leave whole column as-is unconverted

    return df


def _read_curves(fn):
    # open SwapCurve files (based on R S3 class Discount Curves) and convert to
    # dictionary with nested arrays, dictionaries and one nested dataframe

    with open(fn) as f:
        dd = _json.load(f)

    for key in dd.keys():
        if isinstance(dd[key], list) == True:
            dd[key] = _np.array(dd[key])

        if isinstance(dd[key], dict) == True:
            # print(dd[key])
            for k in dd[key].keys():
                dd[key][k] = _np.array(dd[key][k])

                if len(dd[key][k]) == 1:
                    dd[key][k] = dd[key][k][0]


    dd["table"] = _pd.DataFrame.from_records(dd["table"])

    dd["table"]["date"] = _pd.to_datetime(dd["table"]["date"])  # , utc=True, unit="D"

    return dd


def _read_dict(fn):
    with open(fn) as f:
        dd = _json.load(f)

    return dd


_file_actions = {
    "crudeOil": {
        "file": "crudeOil.json",
        "date_fields": None,
        "load_func": _norm_df,
    },
    "cushing": {
        "file": "cushing.json",
        "date_fields": None,
        "load_func": _load_data,
    },
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
    "eurodollar": {
        "file": "eurodollar.json",
        "date_fields": None,
        "load_func": _load_data,
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
    "futuresRef": {
        "file": "futuresRef.json",
        "date_fields": None,
        "load_func": _norm_df,
    },
    "fxfwd": {
        "file": "fxfwd.json",
        "date_fields": None,
        "load_func": _load_data,
    },
    "holidaysOil": {
        "file": "holidaysOil.json",
        "date_fields": ["value"],
        "load_func": _pd.read_json,
    },
    "ohlc": {
        "file": "ohlc.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "planets": {
        "file": "planets.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "refineryLPdata": {
        "file": "refineryLPdata.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
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
    "tradeHubs": {
        "file": "tradeHubs.json",
        "date_fields": None,
        "load_func": _pd.read_json,
    },
    "tradeprocess": {
        "file": "tradeprocess.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "wti_swap": {
        "file": "wtiSwap.json",
        "date_fields": ["date"],
        "load_func": _pd.read_json,
    },
    "stocks": {
        "file": "stocks.json",
        "date_fields": None,
        "load_func": _load_data,
    },
    "tsQuotes": {
        "file": "tsQuotes.json",
        "date_fields": None,
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
}


_path = _os.path.dirname(__file__)


if __name__ == "__main__":
    pass
