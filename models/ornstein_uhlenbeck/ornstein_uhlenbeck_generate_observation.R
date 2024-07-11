source("models/ornstein_uhlenbeck/ornstein_uhlenbeck_model.R")

set.seed(1)
true_param = c(3, 1, 0.5)
observation = model(true_param, model_param)

plot(observation, type = "b") # plot observation

# write observation to file
write.table(
  observation,
  file = "models/ornstein_uhlenbeck/num_observation_1/observation.csv",
  sep = ",",
  row.names = F,
  col.names = F
)