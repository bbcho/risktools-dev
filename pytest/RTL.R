# Script to building output files from each RTL function to use for testing
# purposes in risktools

library(PerformanceAnalytics)
library(tidyverse)
library(tidyquant)
library(RTL)
library(timetk)
library(jsonlite)

# Give the input file name to the function.
up <- jsonlite::fromJSON(file("../../../user.json"))

username <- up$'m*'$user
password <- up$'m*'$pass
eia_key = up$eia

# ir_df_us
## Must be run on the same day as the data pull since the function only gets the last
## 30 days
df <- ir_df_us()
df <- jsonlite::toJSON(df)
write(df,'ir_df_us.json')


# CRReuro
df <- CRReuro(S=100,X=100,sigma=0.2,r=0.1,T2M=1,N=5,type="call")
df <- jsonlite::toJSON(df)
write(df,'crreuro.json')

# bond
ou1 <- bond(ytm = 0.05, C = 0.05,T2M = 1,m = 2,output = "price")
ou2 <- bond(ytm = 0.05,C = 0.05,T2M = 1,m = 2,output = "df")
ou3 <- bond(ytm = 0.05,C = 0.05,T2M = 1,m = 2,output = "duration")

write(jsonlite::toJSON(ou1), 'bond_1.json')
write(jsonlite::toJSON(ou2), 'bond_2.json')
write(jsonlite::toJSON(ou3), 'bond_3.json')


# chart_eia_sd
ou <- chart_eia_sd(key = eia_key, market = "mogas", output='data')
write(jsonlite::toJSON(ou), 'chart_eia_sd.json')


# chart_eia_steo
ou <- chart_eia_steo(key = eia_key, market = "globalOil", output='data')
write(jsonlite::toJSON(ou), 'chart_eia_steo.json')


# chart_spreads
cpairs <- dplyr::tibble(year = c("2014","2019","2020"),
first = c("@HO4H","@HO9H","@HO0H"),
second = c("@HO4J","@HO9J","@HO0J"))
ou <- chart_spreads(cpairs = cpairs, daysFromExpiry = 200, from = "2012-01-01",
conversion = 42,feed = "CME_NymexFutures_EOD",
iuser = username, ipassword = password,
title = "March/April ULSD Nymex Spreads",
yaxis = "$ per bbl",
output = "data")
write(jsonlite::toJSON(ou), 'chart_spreads.json')


# chart_zscore
df <- eiaStocks %>% dplyr::filter(series == "NGLower48")
title <- "NGLower48"
ou <- chart_zscore(df = df, title = " ",per = "yearweek", output = "res", chart = "seasons")
ou <- as_tibble(ou) %>% select(c(freq,value,trend,season_year,remainder,season_adjust))
ou$freq <- as.Date(ou$freq)
write(jsonlite::toJSON(ou), 'chart_zscore.json')

# eia2tidy
ou1 <- RTL::eia2tidy(ticker = "PET.MCRFPTX2.M", key = eia_key, name = "TexasProd")
write(jsonlite::toJSON(ou1), 'eia2tidy1.json')

ou2 <-tibble::tribble(~ticker, ~name,
"PET.W_EPC0_SAX_YCUOK_MBBL.W", "CrudeCushing",
"NG.NW2_EPG0_SWO_R48_BCF.W","NGLower48") %>%
dplyr::mutate(key = eia_key) %>%
dplyr::mutate(df = purrr::pmap(list(ticker,key,name),.f=RTL::eia2tidy)) %>%
dplyr::select(df) %>% tidyr::unnest(df)
write(jsonlite::toJSON(ou2), 'eia2tidy2.json')


# garch
x <- dflong %>% dplyr::filter(series=="CL01")
x <- returns(df=x,retType="rel",period.return=1,spread=TRUE)
x <- rolladjust(x=x,commodityname=c("cmewti"),rolltype=c("Last.Trade"))
summary(garch(x=x,out="fit"))
ou <- garch(x=x,out="data")
write(jsonlite::toJSON(ou), 'garch.json')


# morningstar
ou1 <- getPrice(feed="CME_NymexFutures_EOD",contract="@CL0Z",
from="2019-08-26",iuser = username, ipassword = password)

ou2 <- getPrice(feed="CME_NymexOptions_EOD",contract="@LO21ZP4000",
from="2020-03-15",iuser = username, ipassword = password)

ou3 <- getPrice(feed="CME_CbotFuturesEOD",contract="C0Z",
from="2019-08-26",iuser = username, ipassword = password)

ou4 <- getPrice(feed="CME_CmeFutures_EOD_continuous",contract="HE_006_Month",
from="2019-08-26",iuser = username, ipassword = password)

ou5 <- getPrice(feed="Morningstar_FX_Forwards",contract="USDCAD 2M",
from="2019-08-26",iuser = username, ipassword = password)

ou6 <- getPrice(feed="ICE_EuroFutures",contract="BRN0Z",
from="2019-08-26",iuser = username, ipassword = password)

ou7 <- getPrice(feed="ICE_NybotCoffeeSugarCocoaFutures",contract="SB21H",
from="2019-08-26",iuser = username, ipassword = password)

ou8 <- getPrice(feed="ICE_NybotCoffeeSugarCocoaFutures_continuous",contract="SF_001_Month",
from="2019-08-26",iuser = username, ipassword = password)

write(jsonlite::toJSON(ou1), 'morningstar1.json')
write(jsonlite::toJSON(ou2), 'morningstar2.json')
write(jsonlite::toJSON(ou3), 'morningstar3.json')
write(jsonlite::toJSON(ou4), 'morningstar4.json')
write(jsonlite::toJSON(ou5), 'morningstar5.json')
write(jsonlite::toJSON(ou6), 'morningstar6.json')
write(jsonlite::toJSON(ou7), 'morningstar7.json')
write(jsonlite::toJSON(ou8), 'morningstar8.json')

# npv
us.df <- ir_df_us(ir.sens=0.01)
npv(init.cost=-375,C=50,cf.freq=.5,TV=250,T2M=2,
disc.factors=us.df,BreakEven=TRUE,BE.yield=.0399)$npv

npv(init.cost=-375,C=50,cf.freq=.5,TV=250,T2M=2,
disc.factors=us.df,BreakEven=TRUE,BE.yield=.0399)$df


# promptBeta
x <- dflong %>% dplyr::filter(grepl("CL",series))
x <- x %>% dplyr::mutate(series = readr::parse_number(series)) %>% dplyr::group_by(series)
x <- RTL::returns(df = x,retType = "abs",period.return = 1,spread = TRUE)
x <- RTL::rolladjust(x = x,commodityname = c("cmewti"),rolltype = c("Last.Trade"))
x <- x %>% dplyr::filter(!grepl("2020-04-20|2020-04-21",date))
ou <- promptBeta(x = x,period = "all",betatype = "all",output = "betas")
write(jsonlite::toJSON(ou), 'promptBeta.json')


# returns
ou1 <- returns(df=dflong,retType="rel",period.return=1,spread=TRUE)
ou2 <- returns(df=dflong,retType="rel",period.return=1,spread=FALSE)
write(jsonlite::toJSON(ou1), 'returns1.json')
write(jsonlite::toJSON(ou2), 'returns2.json')


# rolladjust
ret <- returns(df=dflong,retType="abs",period.return=1,spread=TRUE)[,1:2] 
ou <- rolladjust(x=ret,commodityname=c("cmewti"),rolltype=c("Last.Trade"))
write(jsonlite::toJSON(ou), 'rolladjust.json')


# stl_decomp
x <- dflong %>% dplyr::filter(series=="CL01")
ou <- stl_decomp(x,output="data",s.window=13,s.degree=1)
write(jsonlite::toJSON(ou$Components), 'stl_decomp.json')


# swapCom
c <- paste0("CL0",c("M","N","Q"))
futs <-getPrices(feed="CME_NymexFutures_EOD",contracts = c,from="2019-08-26",
iuser = username, ipassword = password)
ou <- swapCOM(futures = futs, futuresNames=c("CL0M","CL0N"),
pricingDates = c("2020-05-01","2020-05-30"), contract = "cmewti", exchange = "nymex")
write(jsonlite::toJSON(ou), 'swapCOM.json')


# swapIRS
ou <- swapIRS(trade.date = as.Date("2020-01-04"), eff.date = as.Date("2020-01-06"),
mat.date = as.Date("2022-01-06"), notional = 1000000,
PayRec = "Rec", fixed.rate=0.05, float.curve = usSwapCurves, reset.freq=3,
disc.curve = usSwapCurves, convention = c("act",360),
bus.calendar = "NY", output = "all")
write(jsonlite::toJSON(ou), 'swapIRS.json')


# tradeStats
library(quantmod)
getSymbols("SPY", return.class = "zoo")
SPY$retClCl <- na.omit(quantmod::Delt(Cl(SPY),k=1,type='arithmetic'))
ou <- tradeStats(x=SPY$retClCl,Rf=0)
write(jsonlite::toJSON(ou), 'tradeStats.json')
