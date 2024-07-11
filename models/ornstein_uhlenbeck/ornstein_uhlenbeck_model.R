#::::::::::::::::::::::::::::::: MODEL PARAMETERS ::::::::::::::::::::::::::::::
T=10
n=50
model_param = list(T=T, n=n)
dim_data = n+1
num_param = 3

#::::::::::::::::::::::::::::: GENERATIVE MODEL ::::::::::::::::::::::::::::::::
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

#:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
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

#:::::::::::::::::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::::::::::
jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

# ::::::::::::::::::: METROPOLIS HASTINGS HELP FUNCTIONS :::::::::::::::::::::::
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