#:::: Generative model ::::::::::::::::::::::::::::::::

model = function(param, model_param=NULL){
  # list2env(model_param, environment()) # import model parameter list (T, n) to function environment
  T = model_param$T
  n = model_param$n
  delta = T/n
  
  alpha = param[1]
  beta = param[2]
  sigma = param[3]
  
  x0 = 0
  x = rep(NA, n+1)
  x[1] = x0
  for (i in 2:(n+1)){
    x[i] =  alpha + (x[i-1]-alpha)*exp(-beta*delta) + sqrt((sigma^2/(2*beta))*(1-exp(-2*beta*delta))) * rnorm(1)
  }
  return(x)
}

#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta=theta){
  alpha = theta[1]
  beta = theta[2]
  sigma = theta[3]
  
  jointprior = dunif(alpha,0,10)*dunif(beta,0,5)*dunif(sigma,0,2)
  return(jointprior)
}

sample_prior = function(){
  alpha = runif(1, min=0, max=10)
  beta = runif(1, min=0, max=5)
  sigma = runif(1, min=0, max=2)
  
  return(c(alpha, beta, sigma))
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}
# ::::::::::::::::::: METROPOLIS HASTINGS HELP FUNCTIONS :::::::::::::::::::::
loglik_func = function(x, alpha, beta, sigma, delta){
  loglik = 0
  for (i in 2:length(x)){
      mean = alpha + (x[i-1]-alpha)*exp(-beta*delta)
      sd = sqrt((sigma^2/(2*beta))*(1-exp(-2*beta*delta)))
      loglik = loglik + dnorm(x[i], mean=mean, sd=sd, log=TRUE)
  }
  return(loglik)
}

log_post_unnormalized = function(x, param, T){
  alpha = param[1]
  beta = param[2]
  sigma = param[3]
  delta = T/(length(x)-1)
  
  if((alpha<0) || (alpha>10) || (beta<0) || (beta>5) || (sigma<0) || (sigma>3)){
    return(-Inf)
  }else{
    return(loglik_func(x, alpha, beta, sigma, delta))
  }
}

#::::::::::::::::::::::::: MODEL PARAMETERS :::::::::::::::::::::::::

T=10
n=50
model_param = list(T=T, n=n)
dim_data = n+1
num_param = 3
# true_param = c(3,1,0.5)

# # Generate new observation
# set.seed(1)
# observation = model(true_param, model_param)
# plot(observation, type="b")
# write.table(observation, file = "results/ornstein_uhlenbeck/observation.dat", sep = ",", row.names = F, col.names = F)



#:::::::::::::::::::::::::::::: RUN METROPOLIS HASTINGS::::::::::::::::::::::::::::::::::
# library(mcmc)
# observedData = read.table(file="results/ornstein_uhlenbeck/observation.dat")
# set.seed(1)
# metrop_output = metrop(log_post_unnormalized, initial=c(3,1,0.5), blen = 1, nbatch = 10000, nspac=10, x=unlist(observedData), T=T, scale=0.3)
# MH_values = metrop_output$batch
# plot(MH_values[,1], type="l", ylab="alpha")
# plot(MH_values[,2], type="l", ylab="beta")
# plot(MH_values[,3], type="l", ylab="sigma")
# ggpairs(data.frame(MH_values), diag=list(continuous="barDiag"), upper=list(continuous="points"))
# 
# write.table(MH_values, file = paste("results/ornstein_uhlenbeck/MH_sample_10k.dat", sep = ''), sep = "\t", row.names = F, col.names = F)
# write.table(MH_values, file = paste("results/ornstein_uhlenbeck/MH_sample_10k.csv", sep = ''), sep=",", row.names = F, col.names = F)







