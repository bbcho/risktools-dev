# Script to building output files from each RTL function to use for testing
# purposes in risktools

library(PerformanceAnalytics)
library(tidyverse)
library(tidyquant)
library(RTL)
library(timetk)
library(jsonlite)

# Give the input file name to the function.
up <- jsonlite::fromJSON(file("../../user.json"))

username <- up$"m*"$user
password <- up$"m*"$pass
eia_key <- up$eia
q_key <- Sys.getenv("QUANDL_KEY")


# get_prices
from_dt = "2021-12-01"
end_dt = format(Sys.Date(), "%Y-%m-%d")
df <- list()
df[[1]] <- c(feed="CME_NymexFutures_EOD", contract="@CL22Z", from=from_dt, end=end_dt, df=list())
df[[2]] <- c(feed="CME_NymexFutures_EOD_continuous",contract="CL_006_Month",from=from_dt, end=end_dt, df=list())
df[[3]] <- c(feed="CME_CbotFuturesEOD",contract="C0Z",from='2020-09-01', end=end_dt, df=list())
df[[4]] <- c(feed="CME_CbotFuturesEOD_continuous",contract="ZB_001_Month",from=from_dt,end=end_dt, df=list())
df[[5]] <- c(feed="Morningstar_FX_Forwards",contract="USDCAD 2M",from=from_dt,end=end_dt, df=list())
df[[6]] <- c(feed="CME_CmeFutures_EOD",contract="LH0N",from="2019-08-26",end=end_dt, df=list())
df[[7]] <- c(feed="CME_CmeFutures_EOD_continuous",contract="HE_006_Month",from=from_dt,end=end_dt, df=list())
df[[8]] <- c(feed="ICE_EuroFutures",contract="BRN22Z",from=from_dt,end=end_dt, df=list())
df[[9]] <- c(feed="ICE_EuroFutures_continuous",contract="BRN_001_Month",from=from_dt,end=end_dt, df=list())
df[[10]] <- c(feed="ICE_NybotCoffeeSugarCocoaFutures",contract="SB21H",from='2019-08-26',end=end_dt, df=list())
df[[11]] <- c(feed="ICE_NybotCoffeeSugarCocoaFutures_continuous",contract="SF_001_Month",from=from_dt,end=end_dt, df=list())
df[[12]] <- c(feed="LME_MonthlyDelayed_Derived",contract="AHD 2021-12-01 2021-12-31",from=from_dt,end=end_dt, df=list())
# df[[13]] <- c(feed="AESO_ForecastAndActualPoolPrice",contract="Forecast_Pool_Price",from=from_dt,end=end_dt, df=list())

for (i in 1:length(df)) {
  df[[i]]$df <- RTL::getPrice(feed=df[[i]]$feed,contract=df[[i]]$contract,from=df[[i]]$from,iuser=username,ipassword=password)
}
write(jsonlite::toJSON(df, digits = 8), "get_price.json")


#swapInfo

feeds = dplyr::tibble(
  feed = c("Crb_Futures_Price_Volume_And_Open_Interest",
           "CME_NymexFutures_EOD_continuous"),
  ticker = c("CL","CL_001_Month")
)
df <- swapInfo(date = "2020-05-06",feeds = feeds, contract = "cmewti",exchange = "nymex",
                    iuser = username, ipassword = password, output = "all")[[1]]
write(jsonlite::toJSON(df, digits = 8), "swapInfo.json")

# getCurve
df <- getCurve(iuser=username, ipass=password, date='2021-12-20', contract="CL")
write(jsonlite::toJSON(df, digits = 8), "getCurveCL.json")
df <- getCurve(iuser=username, ipass=password, date='2021-12-20', contract="BG")
write(jsonlite::toJSON(df, digits = 8), "getCurveBG.json")

# rewrite because of bug in getPrices
getPrices <- function(feed = "CME_NymexFutures_EOD",contracts = c("CL9Z","CL0F","CL0M"),from = "2019-01-01",iuser = "x@xyz.com", ipassword = "pass") {
  x <- getPrice(feed=feed,contract=contracts[1],from=from,iuser = iuser, ipassword = ipassword)
  for (c in contracts[-1]) {
    x <- merge(x,getPrice(feed=feed,contract=c,from=from,iuser = iuser, ipassword = ipassword), all=TRUE)
  }
  
  x <- dplyr::as_tibble(x)
  return(x)
}

getIRswapCurve <- function(currency="USD",from="2019-01-01",iuser = "x@xyz.com", ipassword = "pass") {
  
  usSwapIR <- dplyr::tibble(tickQL = c("d1d","d1w","d1m","d3m","d6m","d1y",
                                       paste0("fut",1:8),
                                       paste0("s",c(2,3,5,7,10,15,20,30),"y")),
                            type = c(rep("ICE.LIBOR",6),rep("EuroDollar",8),rep("IRS",8)),
                            source = c(rep("FRED",6),rep("Morningstar",8),rep("FRED",8)),
                            tickSource = c("USDONTD156N","USD1WKD156N","USD1MTD156N","USD3MTD156N","USD6MTD156N","USD12MD156N",
                                           paste0("ED_",sprintf('%0.3d', 1:8),"_Month"),
                                           paste0("ICERATES1100USD",c(2,3,5,7,10,15,20,30),"Y")))
  
  c = usSwapIR %>% dplyr::filter(source == "Morningstar") %>% .$tickSource
  r <- getPrices(feed="CME_CmeFutures_EOD_continuous",contracts=c,from = from,iuser = iuser, ipassword = ipassword)
  c = usSwapIR %>% dplyr::filter(source == "FRED") %>% .$tickSource
  x <- tidyquant::tq_get(c, get  = "economic.data", from = from ,to = as.character(Sys.Date())) %>%
    dplyr::mutate(price=price/100) %>%
    tidyr::pivot_wider(date,names_from = symbol, values_from = price)
  r <- dplyr::left_join(x, r, by=c("date"))
  colnames(r) <- c("date",dplyr::tibble(tickSource = colnames(r)[-1]) %>% dplyr::left_join(usSwapIR,by = c("tickSource")) %>% .$tickQL)
  return(r)
}


# getIRswapCurve
df <- getIRswapCurve(iuser=username, ipass=password, )
write(jsonlite::toJSON(df, digits = 8), "getIRSwapCurve.json")


# ir_df_us
## Must be run on the same day as the data pull since the function only gets the last
## 30 days
df <- RTL::ir_df_us(quandlkey = up$"quandl")
write(jsonlite::toJSON(df, digits = 8), "ir_df_us.json")

# npv
## Must be run on the same day as the data pull since the ir_df_us function only gets the last
## 30 days
us.df <- ir_df_us(quandlkey = up$"quandl", ir.sens = 0.01)
ou1 <- npv(
  init.cost = -375, C = 50, cf.freq = .5, TV = 250, T2M = 2,
  disc.factors = us.df, BreakEven = FALSE
)$df
write(jsonlite::toJSON(ou1), "npv1.json")

ou2 <- npv(
  init.cost = -375, C = 50, cf.freq = .5, TV = 250, T2M = 2,
  disc.factors = us.df, BreakEven = TRUE, BE.yield = .0399
)$df
write(jsonlite::toJSON(ou2), "npv2.json")

# CRReuro
df <- CRReuro(S = 100, X = 100, sigma = 0.2, r = 0.1, T2M = 1, N = 5, type = "call")
df <- jsonlite::toJSON(df)
write(df, "crreuro.json")

# bond
ou1 <- bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "price")
ou2 <- bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "df")
ou3 <- bond(ytm = 0.05, C = 0.05, T2M = 1, m = 2, output = "duration")

write(jsonlite::toJSON(ou1), "bond_1.json")
write(jsonlite::toJSON(ou2), "bond_2.json")
write(jsonlite::toJSON(ou3), "bond_3.json")


# chart_eia_sd
ou <- chart_eia_sd(key = eia_key, market = "mogas", output = "data")
write(jsonlite::toJSON(ou), "chart_eia_sd.json")


# chart_eia_steo
ou <- chart_eia_steo(key = eia_key, market = "globalOil", output = "data")
write(jsonlite::toJSON(ou), "chart_eia_steo.json")

cpairs <- dplyr::tibble(
  year = c("2014", "2019", "2020"),
  first = c("@HO4H", "@HO9H", "@HO0H"),
  second = c("@HO4J", "@HO9J", "@HO0J")
)
ou <- chart_spreads(
  cpairs = cpairs, daysFromExpiry = 200, from = "2012-01-01",
  conversion = 42, feed = "CME_NymexFutures_EOD",
  iuser = username, ipassword = password,
  title = "March/April ULSD Nymex Spreads",
  yaxis = "$ per bbl",
  output = "data"
)
write(jsonlite::toJSON(ou), "chart_spreads.json")

# chart_spreads
chart_spreads(
  cpairs = cpairs,
  daysFromExpiry = 200,
  from = "2012-01-01",
  conversion = c(1, 1),
  feed = "CME_NymexFutures_EOD",
  iuser = username,
  ipassword = password,
  title = "March/April ULSD Nymex Spreads",
  yaxis = "$ per bbl",
  output = "chart"
)

cpairs <- dplyr::tibble(
  year = c("2014", "2019", "2020"),
  first = c("@HO4H", "@HO9H", "@HO0H"),
  second = c("@HO4J", "@HO9J", "@HO0J")
)
ou <- chart_spreads(
  cpairs = cpairs, daysFromExpiry = 200, from = "2012-01-01",
  conversion = 42, feed = "CME_NymexFutures_EOD",
  iuser = username, ipassword = password,
  title = "March/April ULSD Nymex Spreads",
  yaxis = "$ per bbl",
  output = "chart"
)

# chart_zscore
df <- eiaStocks %>% dplyr::filter(series == "NGLower48")
title <- "NGLower48"
ou <- chart_zscore(df = df, title = " ", per = "yearweek", output = "res", chart = "seasons")
ou <- as_tibble(ou) %>% select(c(freq, value, trend, season_year, remainder, season_adjust))
ou$freq <- as.Date(ou$freq)
write(jsonlite::toJSON(ou), "chart_zscore.json")

# eia2tidy
ou1 <- RTL::eia2tidy(ticker = "PET.MCRFPTX2.M", key = eia_key, name = "TexasProd")
write(jsonlite::toJSON(ou1), "eia2tidy1.json")

ou2 <- tibble::tribble(
  ~ticker, ~name,
  "PET.W_EPC0_SAX_YCUOK_MBBL.W", "CrudeCushing",
  "NG.NW2_EPG0_SWO_R48_BCF.W", "NGLower48"
) %>%
  dplyr::mutate(key = eia_key) %>%
  dplyr::mutate(df = purrr::pmap(list(ticker, key, name), .f = RTL::eia2tidy)) %>%
  dplyr::select(df) %>%
  tidyr::unnest(df)
write(jsonlite::toJSON(ou2), "eia2tidy2.json")


# garch
x <- dflong %>% dplyr::filter(series == "CL01")
x <- returns(df = x, retType = "rel", period.return = 1, spread = TRUE)
x <- rolladjust(x = x, commodityname = c("cmewti"), rolltype = c("Last.Trade"))
ou <- garch(x = x, out = "data")
ou <- tk_tbl(ou, preserve_index = TRUE, rename_index = "date")
write(jsonlite::toJSON(as_tibble(ou)), "garch.json")


# morningstar
ou1 <- getPrice(
  feed = "CME_NymexFutures_EOD", contract = "@CL0Z",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou2 <- getPrice(
  feed = "CME_NymexOptions_EOD", contract = "@LO21ZP4000",
  from = "2020-03-15", iuser = username, ipassword = password
)

ou3 <- getPrice(
  feed = "CME_CbotFuturesEOD", contract = "C0Z",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou4 <- getPrice(
  feed = "CME_CmeFutures_EOD_continuous", contract = "HE_006_Month",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou5 <- getPrice(
  feed = "Morningstar_FX_Forwards", contract = "USDCAD 2M",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou6 <- getPrice(
  feed = "ICE_EuroFutures", contract = "BRN0Z",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou7 <- getPrice(
  feed = "ICE_NybotCoffeeSugarCocoaFutures", contract = "SB21H",
  from = "2019-08-26", iuser = username, ipassword = password
)

ou8 <- getPrice(
  feed = "ICE_NybotCoffeeSugarCocoaFutures_continuous", contract = "SF_001_Month",
  from = "2019-08-26", iuser = username, ipassword = password
)

write(jsonlite::toJSON(ou1), "morningstar1.json")
write(jsonlite::toJSON(ou2), "morningstar2.json")
write(jsonlite::toJSON(ou3), "morningstar3.json")
write(jsonlite::toJSON(ou4), "morningstar4.json")
write(jsonlite::toJSON(ou5), "morningstar5.json")
write(jsonlite::toJSON(ou6), "morningstar6.json")
write(jsonlite::toJSON(ou7), "morningstar7.json")
write(jsonlite::toJSON(ou8), "morningstar8.json")


# promptBeta
x <- dflong %>% dplyr::filter(grepl("CL", series))
x <- x %>%
  dplyr::mutate(series = readr::parse_number(series)) %>%
  dplyr::group_by(series)
x <- RTL::returns(df = x, retType = "abs", period.return = 1, spread = TRUE)
x <- RTL::rolladjust(x = x, commodityname = c("cmewti"), rolltype = c("Last.Trade"))
x <- x %>% dplyr::filter(!grepl("2020-04-20|2020-04-21", date))
ou <- promptBeta(x = x, period = "all", betatype = "all", output = "betas")
write(jsonlite::toJSON(ou), "promptBeta.json")


# returns
# round dflong to 4 decimals before calc because toJSON only writes
# to 4 decimals
dflong_rd <- dflong
dflong_rd$value <- round(dflong_rd$value, 4)
ou1 <- returns(df = dflong_rd, retType = "rel", period.return = 1, spread = TRUE)
ou2 <- returns(df = dflong_rd, retType = "rel", period.return = 1, spread = FALSE)
ou3 <- returns(df = dflong_rd, retType = "abs", period.return = 1, spread = TRUE)
ou4 <- returns(df = dflong_rd, retType = "log", period.return = 1, spread = TRUE)

write(jsonlite::toJSON(ou1), "returns1.json")
write(jsonlite::toJSON(ou2), "returns2.json")
write(jsonlite::toJSON(ou3), "returns3.json")
write(jsonlite::toJSON(ou4), "returns4.json")


# rolladjust
dfra <- dflong %>% filter(series=='CL01')
ret <- returns(df = dfra, retType = "abs", period.return = 1, spread = TRUE)[, 1:2]
ou <- rolladjust(x = ret, commodityname = c("cmewti"), rolltype = c("Last.Trade"))
write(jsonlite::toJSON(ou), "rolladjust.json")


# stl_decomp
x <- dflong %>% dplyr::filter(series == "CL01")
ou <- stl_decomp(x, output = "data", s.window = 13, s.degree = 1)
ou <- tk_tbl(ou$time.series)
ou$index <- as.Date(ou$index)
write(jsonlite::toJSON(ou), "stl_decomp.json")


# swapCom
c <- paste0("CL0", c("M", "N", "Q"))
futs <- getPrices(
  feed = "CME_NymexFutures_EOD", contracts = c, from = "2019-08-26",
  iuser = username, ipassword = password
)
ou <- swapCOM(
  futures = futs, futuresNames = c("CL0M", "CL0N"),
  pricingDates = c("2020-05-01", "2020-05-30"), contract = "cmewti", exchange = "nymex"
)
write(jsonlite::toJSON(ou), "swapCOM.json")


# swapIRS
ou <- swapIRS(
  trade.date = as.Date("2020-01-04"), eff.date = as.Date("2020-01-06"),
  mat.date = as.Date("2021-12-06"), notional = 1000000,
  PayRec = "Rec", fixed.rate = 0.05, float.curve = usSwapCurves, reset.freq = 3,
  disc.curve = usSwapCurves, convention = c("act", 360),
  bus.calendar = "NY", output = "all"
)
# ou <- paste0('{"v1":',ou[[1]][1],',"df":',jsonlite::toJSON(ou[[2]])[1],',"v2":',ou[[3]][1],'}')
write(jsonlite::toJSON(ou[[2]]), "swapIRS.json")
print(ou[[1]])
print(ou[[3]])


# tradeStats
library(xts)

spy <- tq_get("SPY", from = "2000-01-01", to = "2012-01-01", return.class = "zoo")
spy <- spy$SPY.Adjusted
spy <- spy / xts::lag.xts(spy) - 1
ou <- tradeStats(x = spy, Rf = 0)
write(jsonlite::toJSON(ou), "tradeStats.json")


# alt_garch

alt_garch <- function(x = x, out = TRUE) {
  x <- xts::as.xts(x[, 2], order.by = x$date)
  x <- x - mean(x)
  fit <- rugarch::ugarchfit(data = x, spec = rugarch::ugarchspec(), solver = "hybrid")
  if (xts::periodicity(x)$scale == "daily") {
    garchvol <- fit@fit$sigma * sqrt(252)
  }
  if (xts::periodicity(x)$scale == "weekly") {
    garchvol <- fit@fit$sigma * sqrt(52)
  }
  if (xts::periodicity(x)$scale == "monthly") {
    garchvol <- fit@fit$sigma * sqrt(12)
  }

  voldata <- merge(x, garchvol)
  colnames(voldata) <- c("returns", "garch")
  if (out == "data") {
    return(voldata)
  } else if (out == "fit") {
    return(fit)
  } else {
    xts::plot.xts(voldata[, 2], main = paste("Period Returns and Annualized Garch(1,1) for", colnames(x)[1]), ylim = c(-max(abs(voldata[, 1])), max(abs(voldata))))
    xts::addSeries(voldata[, 1], type = "h", col = "blue")
  }
}

x <- dflong %>% dplyr::filter(series == "CL01")
x <- returns(df = x, retType = "rel", period.return = 1, spread = TRUE)
x <- rolladjust(x = x, commodityname = c("cmewti"), rolltype = c("Last.Trade"))
ou <- garch(x = x, out = "data")
ou <- tk_tbl(ou, preserve_index = TRUE, rename_index = "date")
write(jsonlite::toJSON(as_tibble(ou)), "garch.json")



x <- dflong %>% dplyr::filter(grepl("CL", series))
# x <- x %>% dplyr::mutate(series = readr::parse_number(series)) %>% dplyr::group_by(series)
x <- RTL::returns(df = x, retType = "abs", period.return = 1, spread = TRUE)
x <- RTL::rolladjust(x = x, commodityname = c("cmewti"), rolltype = c("Last.Trade"))
x <- x %>% dplyr::filter(!grepl("2020-04-20|2020-04-21", date))