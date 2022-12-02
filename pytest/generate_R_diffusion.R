library(MASS)

set.seed(1)
df <- matrix(stats::rnorm(252 * 20, mean = 0, sd = 1), ncol = 20, nrow = 252)
df
write.matrix(df,file="diffusion.csv", sep=",")

set.seed(1)
df <- RTL::simGBM(nsims=20, S0=10, drift=0.05, sigma=0.2, T2M=1, dt=1/252, vec=T)
df
write_csv(df, "simGBM_output.csv")


