library(xLLiM)
library(mixtools)
library(tictoc)
library(mvtnorm)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set working directory to the location of this script
source('gaumixfit_sbibm.R')
source('models/ornstein_uhlenbeck/ornstein_uhlenbeck_model.R')

# :::::::::: semple PARAMETERS :::::::::::::::::::::::::::::::::::::::
model_name = "ornstein_uhlenbeck"
num_simulation_per_iteration = 10000
num_samples_saved = 10000
num_priorpred_samples = 10000
burnin = 100
num_iters = 4
K_start = 20
cov_structure = ""
factor_cvMH = 1.2
mixprob_thresh = 0.005 #0.01 #0.03
verbose = 1
MH_target = "likelihood"
gllim_maxiter = 300

param_list = list(num_simulation_per_iteration=num_simulation_per_iteration, num_samples_saved=num_samples_saved, num_priorpred_samples=num_priorpred_samples,
                  num_iters=num_iters, burnin=burnin, K_start=K_start, cov_structure=cov_structure, factor_cvMH=factor_cvMH, mixprob_thresh=mixprob_thresh, MH_target=MH_target,
                  gllim_maxiter=gllim_maxiter, T=model_param$T, n=model_param$n)

#:::::::::::::::::::::::::RUN INFERENCE:::::::::::::::::::::::::::::::
observedData = read.csv(file="models/ornstein_uhlenbeck/num_observation_1/observation.csv")
subfolder = "10k_prior/" # e.g. test/

run_indices = 10

tic()
for (run_index in run_indices){
  set.seed(run_index)
  
  output_dir = file.path(getwd(), "results", model_name, "semple", paste(subfolder, "run", run_index, sep=''))
  if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}

  settings_json = rjson::toJSON(param_list)
  write(settings_json, paste(output_dir,"/settings.json", sep=''))
  
  start_time = Sys.time()

  inference = gaumixfit(observedData=as.vector(unlist(observedData)),burnin=burnin,K_start=K_start,cov_structure=cov_structure,maxiter=gllim_maxiter,prior_pdf=prior_pdf,sample_prior=sample_prior,
                        jacobian=jacobian,model=model,num_iters=num_iters,factor_cvMH=factor_cvMH,mixprob_thresh=mixprob_thresh,dim_data=dim_data,verbose=verbose,MH_target=MH_target,
                        model_name=model_name,model_param=model_param,num_samples_saved=num_samples_saved,num_simulation_per_iteration=num_simulation_per_iteration,start_time=start_time,output_dir=output_dir)
}
toc()

