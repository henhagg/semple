library(xLLiM)
library(mixtools)
library(tictoc)
library(foreach)
library(doParallel)
#library(SFSI) # bernoulli GLM read binary
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set working directory to the location of this script

#::::::::::::::::::::::BIC COMPUTATION FUNCTION:::::::::::::::::::::::::::::::

compute_bic = function(model_name, num_priorpred_samples, K_values, cov_structure, gllim_verb){
  tic("total")
  source(paste("models/",model_name,"/",model_name,"_model.R",sep=""))
  
  D = dim_data
  L = num_param
  
  set.seed(1) # for reproducibility
  prior_param = replicate(num_priorpred_samples, sample_prior())
  prior_sims = apply(prior_param, 2, model, model_param=model_param)
  
  bic_values = rep(NA,length(K_values))
  
  # Parallelism setup
  totalCores = detectCores()
  cluster <- makeCluster(totalCores[1]-1) 
  registerDoParallel(cluster)
  
  for(K in K_values){
    print(K)
    tic("gllim")
    mod = gllim(prior_param,prior_sims,in_K=K,maxiter=300,verb=gllim_verb,cstr=list(Sigma=cov_structure))
    toc()
    
    # retrieve MLE parameters
    A = mod$A
    b = mod$b
    Sigma = mod$Sigma
    c = mod$c
    Gamma = mod$Gamma
    prior_mix_prob = mod$pi
    
    tic("loglik")
    loglik = foreach(i = 1:num_priorpred_samples, .combine=cbind) %dopar%{
      liksum = 0
      for(k in 1:K){
        lik = prior_mix_prob[k]+
          mixtools::dmvnorm(prior_sims[,i], mu=A[,,k]%*%prior_param[,i], sigma=Sigma[,,k])+
          mixtools::dmvnorm(prior_param[,i], mu=c[,k], sigma=Gamma[,,k])
        liksum = liksum + lik
      }
      log(liksum)
    }
    toc()
    
    if(cov_structure=="i"){
      num_par_Sigma = 1
      num_par_Gamma = 1
    }else if(cov_structure=="d"){
      num_par_Sigma = D
      num_par_Gamma = L
    }else if(cov_structure==""){
      num_par_Sigma = D*(D+1)/2
      num_par_Gamma = L*(L+1)/2
    }else if(cov_structure=="*"){
      num_par_Sigma = D*(D+1)/(2*K)
      num_par_Gamma = L*(L+1)/(2*K)
    }
    
    k = (K-1) + K*(D*L+D+L+num_par_Gamma+num_par_Sigma)  # number of parameters to be estimated by the model
    n = num_priorpred_samples # number of samples
    
    bic_values[which(K_values==K)] = k*log(n) - 2*sum(loglik)
  }
  toc()
  stopCluster(cluster)
  return(bic_values)
}

save_bic_to_csv = function(K_values, bic_values, model_name, num_priorpred_samples, cov_structure){
  save_dir = file.path(getwd(), "results", model_name, "semple", "bic")
  if (!dir.exists(save_dir)) {dir.create(save_dir, recursive=TRUE)}
  
  write.table(rbind(K_values,bic_values), file = paste(save_dir,"/bic_",model_name,"_priorpred", num_priorpred_samples,
                                                       "_cov", cov_structure, "_Kmin", min(K_values), "_Kmax", max(K_values), ".csv", sep = ''), 
              sep=",", row.names = F, col.names = F)
}

save_bic_plot = function(K_values, bic_values, model_name, num_priorpred_samples, cov_structure){
  save_dir = file.path(getwd(), "figures", model_name, "bic")
  if (!dir.exists(save_dir)) {dir.create(save_dir, recursive=TRUE)}
  
  pdf(file=paste(save_dir,"/bic_", model_name, "_priorpred", num_priorpred_samples, "_cov", cov_structure, "_Kmin", min(K_values), "_Kmax", max(K_values), ".pdf",sep=''))
  plot(K_values, bic_values, type="b", xlab="K", ylab="BIC")
  dev.off()
}

plot_bic_from_file = function(model_name, num_priorpred_samples, cov_structure, K_min, K_max){
  bic_table = read.csv(file=paste("results/",model_name,"/semple/bic/bic_",model_name,"_priorpred",num_priorpred_samples,"_cov",cov_structure,"_Kmin",K_min,"_Kmax",K_max,".csv",sep=""), header=FALSE)
  print(bic_table)
  plot(unlist(bic_table[1,]), unlist(bic_table[2,]), type='b')
}


#::::::::::::: TWO MOONS ::::::::::::::::::::::::::::::
cov_structure = ""
model_name = "two_moons"
num_priorpred_samples = 2500

K_values = seq(10,70,10)
bic_values = compute_bic(model_name=model_name, num_priorpred_samples=num_priorpred_samples, K_values=K_values, cov_structure=cov_structure, gllim_verb=0)
plot(K_values, bic_values, type="b", xlab="K", ylab="BIC")
save_bic_to_csv(K_values, bic_values, model_name, num_priorpred_samples, cov_structure)
save_bic_plot(K_values, bic_values, model_name, num_priorpred_samples, cov_structure)

#::::::::::::: MULTIPLE HYPERBOLOID ::::::::::::::::::::::::::::::
cov_structure = "i"
model_name = "hyperboloid"
num_priorpred_samples = 10000

K_values = seq(10,50,10)
bic_values = compute_bic(model_name=model_name, num_priorpred_samples=num_priorpred_samples, K_values=K_values, cov_structure=cov_structure, gllim_verb=0)
plot(K_values, bic_values, type="b", xlab="K", ylab="BIC")
save_bic_to_csv(K_values, bic_values, model_name, num_priorpred_samples, cov_structure)
save_bic_plot(K_values, bic_values, model_name, num_priorpred_samples, cov_structure)


