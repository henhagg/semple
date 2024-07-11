library(xLLiM)
library(mixtools)
library(rjson)
library(smfsb) # for the model simulator

model_name = "lotka_volterra"

source('src/semple.R')
source(paste('models/', model_name, '/', model_name, '_model.R', sep=""))

# :::::::::: Parameters ::::::::::::::::::::

num_simulation_per_iteration = 10000
num_samples_saved = 10000
num_priorpred_samples = 10000

burnin = 100
num_iters = 3
K_start = 15
cov_structure = ""
factor_cvMH = 1
mixprob_thresh = 0.005
verbose = 1
MH_target = "likelihood"
gllim_maxiter = 300

#:::::::::::::::::::::::::RUN INFERENCE:::::::::::::::::::::::::::::::
observation_index = 1
run_indices = 1:5

param_list = list(num_simulation_per_iteration=num_simulation_per_iteration, num_samples_saved=num_samples_saved, 
                  num_priorpred_samples=num_priorpred_samples, num_iters=num_iters, burnin=burnin, K_start=K_start,
                  cov_structure=cov_structure, factor_cvMH=factor_cvMH, mixprob_thresh=mixprob_thresh, 
                  MH_target=MH_target, gllim_maxiter=gllim_maxiter, observation_index=observation_index)

# Read observationv
lv_data = read.csv(paste("models/", model_name, "/num_observation_", observation_index, "/observation_sd30.csv", sep=""), header=FALSE)
observed_summaries = (ssi(lv_data)-means_trim)/sds_trim

subfolder = "test/" # e.g test/

for (run_index in run_indices){
  set.seed(run_index)
  
  output_dir = file.path(getwd(), "results", model_name, "semple", paste(subfolder, "obs", observation_index, "/run", run_index, sep=''))
  if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}
  
  start_time = Sys.time()
  
  settings_json = rjson::toJSON(param_list)
  write(settings_json, paste(output_dir,"/settings.json", sep=''))
  
  inference = semple(observedData=observed_summaries,burnin=burnin,K_start=K_start,cov_structure=cov_structure, maxiter=gllim_maxiter,
                        prior_pdf=prior_pdf,sample_prior=sample_prior,jacobian=jacobian,model=model,num_iters=num_iters,
                        factor_cvMH=factor_cvMH,mixprob_thresh=mixprob_thresh,dim_data=dim_data,verbose=verbose,MH_target=MH_target,
                        model_name=model_name,model_param=model_param,num_samples_saved=num_samples_saved,
                        num_simulation_per_iteration=num_simulation_per_iteration,start_time=start_time,output_dir=output_dir)
}