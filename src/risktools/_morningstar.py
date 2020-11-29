import pandas as _pd
import requests as _requests
import numpy as _np
import json as _json
import urllib as _urllib
import re as _re
import io as _io


def get_prices(
    username,
    password,
    feed="CME_NymexFutures_EOD",
    codes=["CL9Z", "CL0F", "CL0M"],
    start_dt="2019-01-01",
    end_dt=None,
    intraday=False,
):
    """
    function to get prices from the Morningstar data API. The following feeds are supported

        * CME_CbotFuturesEOD and CME_CbotFuturesEOD_continuous
        * CME_NymexFutures_EOD and CME_NymexFutures_EOD_continuous
        * CME_NymexOptions_EOD
        * CME_CmeFutures_EOD and CME_CmeFutures_EOD_continuous
        * CME_Comex_FuturesSettlement_EOD and CME_Comex_FuturesSettlement_EOD_continuous
        * LME_AskBidPrices_Delayed
        * SHFE_FuturesSettlement_RT
        * ICE_EuroFutures and ICE_EuroFutures_continuous
        * ICE_NybotCoffeeSugarCocoaFutures and ICE_NybotCoffeeSugarCocoaFutures_continuous
        * CME_STLCPC_Futures
        * CFTC_CommitmentsOfTradersCombined. Requires multiple keys. Separate them by a space e.g. "N10 06765A NYME 01".
        * Morningstar_FX_Forwards. Requires multiple keys. Separate them by a space e.g. "USDCAD 2M".

    Parameters
    ----------
    username : str
        Morningstar API username
    password : str
        Morningstar API password
    feed : str
        API feed to get data from, by default "CME_NymexFutures_EOD"
    codes : list[tuple[str]]
        either a string ticker code or a list of ticker codes or a futures contract type to return, by default ["CL9Z", "CL0F", "CL0M"]. 
    start_dt : str | datetime, optional
        earliest date to return data from, by default "2019-01-01"
    end_dt : str | datetime, optional
        lastest date to return data from, by default None. If None, the function will use today's date
    intraday : bool
        not implemented yet
    """
    if isinstance(codes, list) == False:
        codes = [codes]
    s = _requests.Session()
    s.auth = (username, password)

    start_dt = _pd.to_datetime(start_dt)

    url = "https://mp.morningstarcommodity.com/lds/feeds/{}/ts?{}"

    df = _pd.DataFrame()

    for code in codes:
        p_dict = dict()
        p_dict["fromDateTime"] = start_dt.strftime("%Y-%m-%d")

        if feed in [
            "CME_NymexFutures_EOD",
            "CME_NymexOptions_EOD",
            "CME_CbotFuturesEOD",
            "CME_CmeFutures_EOD",
            "ICE_EuroFutures",
            "ICE_NybotCoffeeSugarCocoaFutures",
            "CME_Comex_FuturesSettlement_EOD",
            "LME_AskBidPrices_Delayed",
            "SHFE_FuturesSettlement_RT",
        ]:
            p_dict["Symbol"] = code
        elif feed in [
            "CME_NymexFutures_EOD_continuous",
            "CME_CmeFutures_EOD_continuous",
            "ICE_EuroFutures_continuous",
            "ICE_NybotCoffeeSugarCocoaFutures_continuous",
            "CME_Comex_FuturesSettlement_EOD_continuous",
            "CME_CbotFuturesEOD_continuous",
        ]:
            p_dict["Contract"] = code
        elif feed in ["CME_STLCPC_Futures"]:
            p_dict["product"] = code
        elif feed in ["CFTC_CommitmentsOfTradersCombined"]:
            fcode = _re.sub("[^\w]", " ", code).split()  # get rid of any punctuation
            p_dict["cftc_subgroup_code"] = fcode[0]
            p_dict["cftc_contract_market_code"] = fcode[1]
            p_dict["cftc_market_code"] = fcode[2]
            p_dict["cftc_region_code"] = fcode[3]
            p_dict["cols"] = fcode[4]
        elif feed in ["Morningstar_FX_Forwards"]:
            fcode = _re.sub("[^\w]", " ", code).split()  # get rid of any punctuation
            p_dict["cross_currencies"] = fcode[0]
            p_dict["period"] = fcode[1]
        else:
            raise ValueError("feed not recognized")

        if end_dt is not None:
            end_dt = _pd.to_datetime(end_dt)

        params = _urllib.parse.urlencode(p_dict)
        r = s.get(url.format(feed, params),)
        tf = _pd.read_csv(_io.StringIO(r.content.decode("utf-8"))).set_index("Date")
        tf.columns = tf.columns.str.replace(
            "\(([^)]*)\)", ""
        )  # clean up columns by removing ticker/code names
        tf = _pd.concat([tf], keys=[code])

        df = df.append(tf)

    return df


if __name__ == "__main__":
    import json

    with open("../../user.json") as jfile:
        userfile = jfile.read()

    up = json.loads(userfile)

    username = up["m*"]["user"]
    password = up["m*"]["pass"]
    # df = get_prices(
    #     username=username,
    #     password=password,
    #     feed="CME_NymexFutures_EOD",
    #     codes=["@CL0Z", "@CL21Z"],
    #     start_dt="2019-08-26",
    # )

    # df = get_prices(feed="CME_NymexFutures_EOD",codes="@CL0Z",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="CME_NymexFutures_EOD",codes="@CL21Z",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(
    #     feed="CME_NymexFutures_EOD_continuous",
    #     codes=["CL_006_Month", "CL_007_Month"],
    #     start_dt="2019-08-26",
    #     username=username,
    #     password=password,
    # )
    # df = get_prices(feed="CME_NymexOptions_EOD",codes="@LO21ZP4000",start_dt="2020-03-15",username = username, password = password)
    # df = get_prices(feed="CME_CbotFuturesEOD",codes="C0Z",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="CME_CbotFuturesEOD_continuous",codes="ZB_001_Month",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="CME_CmeFutures_EOD_continuous",codes="HE_006_Month",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="Morningstar_FX_Forwards",codes="USDCAD 2M",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="CME_CmeFutures_EOD",codes="LH0N",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="CME_CmeFutures_EOD_continuous",codes="HE_006_Month",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="ICE_EuroFutures",codes="BRN0Z",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="ICE_EuroFutures_continuous",codes="BRN_001_Month",start_dt="2019-08-26",username = username, password = password)
    # df = get_prices(feed="ICE_NybotCoffeeSugarCocoaFutures",codes="SB21H",start_dt="2019-08-26",username = username, password = password)
    df = get_prices(
        feed="ICE_NybotCoffeeSugarCocoaFutures_continuous",
        codes="SF_001_Month",
        start_dt="2019-08-26",
        username=username,
        password=password,
    )

    print(df)
