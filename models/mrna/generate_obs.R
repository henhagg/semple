library(Rcpp)

generate_observations = function(model_name, true_param, observation_index, save_to_file=FALSE){
  source(paste("models/", model_name, "/", model_name, "_model.R", sep=""))
  
  set.seed(observation_index)
  
  observation = model(true_param)
  
  plot(observation, type="l")
  
  if(save_to_file){
    output_dir = paste("models/", model_name, "/num_observation_",observation_index,"/", sep="")
    if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}
    
    write.table(observation, file = paste(output_dir, "observation.csv", sep=""), sep = ",", row.names = F, col.names = F)
    write.table(true_param, paste(output_dir, "true_parameters.csv", sep=""), sep = ",", row.names = F, col.names = F)
  }
}

### settings
model_name = "mrna"

#                   delta,  gamma,     k,     m_0,  scale,    t_0, offset,  sigma
# true_param_1 = c(-0.694, -3.000,  0.027,  5.704,  0.751, -0.164,  2.079, -2.000) # observation 1
# true_param_2 = c(-1.827, -3.138, -0.426,  6.869,  0.697,  0.465,  2.787, -1.993) # observation 2
# true_param_3 = c(-1.153, -3.501, -0.181,  4.601,  0.386, -2.121,  2.185, -2.578) # observation 3
# true_param_4 = c(-1.297, -2.134,  1.118,  5.577,  0.938,  0.120,  1.122, -2.854) # observation 4
# true_param_5 = c(-0.243, -4.596,  0.168,  5.573, -0.086,  1.234,  1.392, -1.655) # observation 5

### run function
# generate_observations(model_name, true_param_1, observation_index=1, save_to_file=F)
# generate_observations(model_name, true_param_2, observation_index=2, save_to_file=T)
# generate_observations(model_name, true_param_3, observation_index=3, save_to_file=T)
# generate_observations(model_name, true_param_4, observation_index=4, save_to_file=T)
# generate_observations(model_name, true_param_5, observation_index=5, save_to_file=T)



