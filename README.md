# risktools

Python implementation of the R package RTL.  

See CRAN for original R version

https://cran.r-project.org/web/packages/RTL/index.html

Purpose

    Purposely designed functions for trading, trading analytics and risk practitioners in Commodities and Finance.
    Build to support delivery of Finance classes from one of the co-authors of RTL at the Alberta School of Business.

Features

    Historical forward curves charting.

    Calendars and expiry dates data objects for a wide range of commodity futures contracts.

    roll_adjust to adjust continuous contracts returns for roll adjustments using expiries above.

    Morningstar Marketplace API functions getPrice(), getPrices() and getCurve() using your own Morningstar credentials. Current feeds included:
        ICE_EuroFutures and ICE_EuroFutures_continuous.
        CME_NymexFutures_EOD and CME_NymexFutures_EOD_continuous.
        CME_NymexOptions_EOD.
        CME_CbotFuturesEOD and CME_CbotFuturesEOD_continuous.
        CME_Comex_FuturesSettlement_EOD and CME_Comex_FuturesSettlement_EOD_continuous.
        LME_AskBidPrices_Delayed.
        CME_CmeFutures_EOD and CME_CmeFutures_EOD_continuous.
        CME_STLCPC_Futures.
        ICE_NybotCoffeeSugarCocoaFutures and ICE_NybotCoffeeSugarCocoaFutures_continuous.
        Morningstar_FX_Forwards.
        … see ?getPrice for up to date selection and examples.

    chart_zscore() supports seasonality adjusted analysis of residuals, particularly useful when dealing with commodity stocks and/or days demand time series with trends as well as non-constant variance across seasonal periods.

    chart_eia_steo() and chart_eia_sd() return either a chart or dataframe of supply demand balances from the EIA.

    chart_spreads() to generate specific contract spreads across years e.g. ULSD March/April. Requires Morningstar credentials.

    swapInfo() returns all information required to price first line futures contract averaging swap or CMA physical trade, including a current month instrument with prior settlements.

Data Sets

Accessible via risktools.data.open_data(datsetname). Also use risktools.data.get_names() to get list of available data.

    expiry_table: Historical and forward futures contract metadata.
    holidaysOil: Holiday calendars for ICE and NYMEX.
    tickers_eia: Mapping of EIA tickers to crude and refined products markets for building supply demand balances.
    usSwapIRDef: Data frame of definitions for instruments to build a curve for use with RQuantlib. Use getIRswapCurve() to extract the latest data from FRED and Morningstar.
    usSwapIR: Sample data set output of getIRswapCurve.
    usSwapCurves: Sample data set output of RQuantlib::DiscountCurve().
    cancrudeassays contains historical Canadian crude assays by batch from Crudemonitor. cancrudeassayssum is a summarised average assays version.
    crudeassaysXOM for all publicly available complete assays in Excel format from ExxonMobil
    crudeassaysBP for all publicly available complete assays in Excel format from BP
    eiaStocks: Sample data set of EIA.gov stocks for key commodiities.
    eiaStorageCap: EIA crude storage capacity by PADD.
    dflong and dfwide contain continuous futures prices sample data sets for Nymex (CL, HO, RB and NG contracts) and ICE Brent.
    crudepipelines and refineries contain GIS information in the North American crude space.

Usernames and password for API services are required.