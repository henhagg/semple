library(xLLiM)
library(mixtools)
library(mvtnorm)
library(rjson)

source('src/semple.R')
source('models/mrna/mrna_model.R')

#:::::::::::::::::: semple parameters ::::::::::::::::::::::::::
model_name = "mrna"
num_simulation_per_iteration = 10000
num_samples_saved = 10000
num_priorpred_samples = 10000
burnin = 100
num_iters = 3
K_start = 10
cov_structure = ""
factor_cvMH = 1
mixprob_thresh = 0.005
verbose = 1
MH_target = "likelihood"
gllim_maxiter = 300

observation_indices = 1:5
subfolder = "test/" # e.g test/

#::::::::::::::::::: RUN INFERENCE ::::::::::::::::::::::::::

for (num_observation in observation_indices){
  set.seed(num_observation+1)
  
  # Read observation
  observation_dir = paste("models/", model_name, "/num_observation_", num_observation, "/", sep="")
  observedData = read.csv(file=paste(observation_dir, "observation.csv", sep=""), header = FALSE)
  true_param_values = read.csv(file=paste(observation_dir, "true_parameters.csv", sep=""), header = FALSE)
  
  # Write settings to file
  param_list = list(num_simulation_per_iteration=num_simulation_per_iteration,
                    num_samples_saved=num_samples_saved,
                    num_priorpred_samples=num_priorpred_samples,
                    num_iters=num_iters,
                    burnin=burnin,
                    K_start=K_start,
                    cov_structure=cov_structure,
                    factor_cvMH=factor_cvMH,
                    mixprob_thresh=mixprob_thresh,
                    MH_target=MH_target,
                    gllim_maxiter=gllim_maxiter,
                    num_param=num_param,
                    dim_data=dim_data,
                    prior_mean=prior_mean,
                    prior_sd=prior_sd,
                    true_param_values=as.vector(unlist(true_param_values)))

  
  output_dir = file.path(getwd(), "results", model_name, "semple", paste(subfolder, "obs", num_observation, sep=''))
  if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}
  
  settings_json = rjson::toJSON(param_list)
  write(settings_json, paste(output_dir,"/settings.json", sep=''))
  
  start_time = Sys.time()
  
  inference = gaumixfit(observedData=as.vector(unlist(observedData)),
                        burnin=burnin,
                        K_start=K_start,
                        cov_structure=cov_structure,
                        maxiter=gllim_maxiter,
                        prior_pdf=prior_pdf,
                        sample_prior=sample_prior,
                        jacobian=jacobian,
                        model=model,
                        num_iters=num_iters,
                        factor_cvMH=factor_cvMH,
                        mixprob_thresh=mixprob_thresh,
                        dim_data=dim_data,
                        verbose=verbose,
                        MH_target=MH_target,
                        model_name=model_name,
                        model_param=model_param,
                        num_samples_saved=num_samples_saved,
                        num_simulation_per_iteration=num_simulation_per_iteration,
                        start_time=start_time,
                        output_dir=output_dir)
}