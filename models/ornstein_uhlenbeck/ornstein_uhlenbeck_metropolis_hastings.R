library(mcmc)
source("models/ornstein_uhlenbeck/ornstein_uhlenbeck_model.R")

#:::::::::::::::::::::::::::: RUN METROPOLIS HASTINGS:::::::::::::::::::::::::::
observedData = read.csv(file = "models/ornstein_uhlenbeck/num_observation_1/observation.csv")
set.seed(1)
num_samples = 10000
true_values = c(3, 1, 0.5)
metrop_output = metrop(
  log_post_unnormalized,
  initial = true_values,
  blen = 1,
  nbatch = num_samples,
  nspac = 10,
  x = unlist(observedData),
  T = T,
  scale = 0.3
)
MH_values = metrop_output$batch

# plot Markov chains
plot(MH_values[,1], type="l", ylab="alpha")
plot(MH_values[,2], type="l", ylab="beta")
plot(MH_values[,3], type="l", ylab="sigma")

# write samples to file
write.table(
  MH_values,
  file = "models/ornstein_uhlenbeck/num_observation_1/MH_sample_10k.csv",
  sep = ",",
  row.names = F,
  col.names = F
)
