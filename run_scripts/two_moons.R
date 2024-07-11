library(xLLiM)
library(mixtools)
library(tictoc)
library(rjson)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set working directory to the location of this script
source('gaumixfit_sbibm.R')
source('models/two_moons/two_moons_model.R')

# :::::::::: Parameters ::::::::::::::::::::
model_name = "two_moons"

num_simulation_per_iteration = 2500
num_samples_saved = 10000
num_priorpred_samples = 2500

burnin = 100
num_iters = 4
K_start = 30
cov_structure = ""
factor_cvMH = 1.2
mixprob_thresh = 0#0.001
verbose = 1
MH_target = "likelihood"
gllim_maxiter = 300

param_list = list(num_simulation_per_iteration=num_simulation_per_iteration, num_samples_saved=num_samples_saved, num_priorpred_samples=num_priorpred_samples,
                  num_iters=num_iters, burnin=burnin, K_start=K_start, cov_structure=cov_structure, factor_cvMH=factor_cvMH, mixprob_thresh=mixprob_thresh, MH_target=MH_target,gllim_maxiter=gllim_maxiter)
subfolder = "full_cov/" # e.g. test/
#:::::::::::::::::::::::::RUN INFERENCE:::::::::::::::::::::::::::::::

tic()
observation_indices = seq(1,10)
for (num_observation in observation_indices){
  set.seed(num_observation)
  
  output_dir = file.path(getwd(), "results", model_name, "semple", paste(subfolder, "obs", num_observation, sep=''))
  if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}
  
  settings_json = rjson::toJSON(param_list)
  write(settings_json, paste(output_dir,"/settings.json", sep=''))
  
  observedData = unlist(read.csv(file=paste("../sbibm/sbibm/tasks/",model_name,"/files/num_observation_", num_observation, "/observation.csv", sep='')))
  
  start_time = Sys.time()
  
  inference = gaumixfit(observedData=observedData,burnin=burnin,K_start=K_start,cov_structure=cov_structure,maxiter=gllim_maxiter,prior_pdf=prior_pdf,sample_prior=sample_prior,jacobian=jacobian,model=model,
                        num_iters=num_iters,factor_cvMH=factor_cvMH,mixprob_thresh=mixprob_thresh,dim_data=dim_data,
                        verbose=verbose,MH_target=MH_target,model_name=model_name,num_samples_saved=num_samples_saved,
                        num_simulation_per_iteration=num_simulation_per_iteration,start_time=start_time,output_dir=output_dir)
}
toc()
