#:::: Generative model ::::::::::::::::::::::::::::::::

model = function(param, model_param=NULL){
  m = c(param[1], param[2])
  
  s1 = param[3]^2
  s2 = param[4]^2
  rho = tanh(param[5])
  
  S = matrix(NA, nrow=2, ncol=2)
  S[1,1] = s1^2
  S[1,2] = rho*s1*s2
  S[2,1] = rho*s1*s2
  S[2,2] = s2^2
  
  eps = 0.000001
  S[1,1] = S[1,1] + eps
  S[2,2] = S[2,2] + eps
  
  simulatedData = mvtnorm::rmvnorm(4, mean=m, sigma=S) # draw 4 two-dimensional points
  
  return(as.vector(t(simulatedData))) # returned flattened 8-dim vector
}


#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta=theta){
  min = -3
  max = 3
  jointprior = dunif(theta[1],min,max)*dunif(theta[2],min,max)*dunif(theta[3],min,max)*dunif(theta[4],min,max)*dunif(theta[5],min,max)
  return(jointprior)
}

sample_prior = function(){
  theta = runif(5, min=-3, max=3)
  return(theta)
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

# :::::::::: Model parameters ::::::::::::::::::::
model_param = NULL

dim_data = 8
num_param = 5





