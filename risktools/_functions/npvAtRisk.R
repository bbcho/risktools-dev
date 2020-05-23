npv.at.risk <- function(init.cost = -400, C = 50, C.cost=1, cf.freq = 0.25, F = 250, T = 2, 
                disc.factors = us.df, BreakEven = FALSE, BE.yield = 0.01,sim.var=FALSE,
                simC=m[,1], X=X) {
  if (BreakEven == TRUE) {
        disc.factors$yield <- BE.yield
        disc.factors <- disc.factors %>% dplyr::mutate(discountfactor = exp(-yield * disc.factors$maturity))
        }
  if(sim.var==FALSE) {
    df <- tibble(t = seq(from = 0, to = T, by = cf.freq), cf = C) %>% 
      dplyr::mutate(cf = case_when(cf >= X  ~ (cf - C.cost), cf < X ~ 0),
                    cf = replace(cf, t == 0, init.cost), cf = replace(cf, t == T, last(cf)+F), 
                    df = spline(x = disc.factors$maturity, y = disc.factors$discountfactor, xout = t)$y, 
                    pv = cf * df)
    }
  if(sim.var==TRUE) {
    df <- tibble(t = seq(from = 0, to = T, by = cf.freq), cf = simC) %>% 
      dplyr::mutate(cf = case_when(cf >= X  ~ (cf - C.cost),cf < X ~ 0),
                    cf = replace(cf, t == 0, init.cost), cf = replace(cf, t == T, last(simC)+F),
                    df = spline(x = disc.factors$maturity, y = disc.factors$discountfactor, xout = t)$y, 
                    pv = cf * df)
    }
  x = list(df = df, npv = sum(df$pv))
  return(x)
}