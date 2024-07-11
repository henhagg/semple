library(mcmc)
source("models/hyperboloid/hyperboloid_model.R")

# Read observation
ytargetITD = read.csv(file="models/hyperboloid/num_observation_1/observation.csv", header=FALSE)

#:::::::::::::::::::::::: RUN METROPOLIS HASTINGS:::::::::::::::::::::::::::::::
set.seed(1)
num_samples = 10000
outITDT = metrop(
  logunpostITDTmix,
  c(0, 0),
  blen = 1,
  nbatch = num_samples,
  nspac = 10,
  yobs = ytargetITD,
  m1 = micr1,
  m2 = micr2,
  m1p = micr1p,
  m2p = micr2p,
  sigmaT = sigmaT,
  dofT = dofT,
  scale = 0.5
)
MHvalITD = outITDT$batch

# write samples to file
write.table(
  MHvalITD,
  file = "models/hyperboloid/num_observation_1/MH_sample.csv",
  sep = ",",
  row.names = F,
  col.names = F
)

# scatter plot samples
dfMHITD = data.frame(MHvalITD)
colnames(dfMHITD) = c('X1', 'X2')
dfconfig = data.frame(matrix(c(-0.5, 0, 0.5, 0, 0, -0.5, 0, 0.5, 1.5, 1), 5, 2, byrow =
                               T))
m = ggplot(dfMHITD, aes(x = X1, y = X2)) +
  theme(aspect.ratio = 1) +
  xlim(-2, 2) +
  ylim(-2, 2)
m = m +  geom_point(
  data = dfMHITD,
  size = 1,
  color = rgb(.1, 0, .9, alpha = 0.5),
  shape = 1
) + xlab("x") + ylab("y")
m = m + geom_point(data = dfconfig,
                   size = 2.5,
                   color = "black") + geom_abline(
                     slope = 0 ,
                     intercept = 0,
                     linetype = "dashed",
                     size = .3
                   )
m

# plot Markov chains
plot(dfMHITD$X1, type='l')
plot(dfMHITD$X2, type='l')