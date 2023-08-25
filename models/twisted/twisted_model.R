#:::: Generative model ::::::::::::::::::::::::::::::::

model = function(param, model_param=NULL){
  
  simulatedData = mvtnorm::rmvnorm(1,mean=param, sigma=diag(length(param))) 
  return(as.vector(t(simulatedData))) 
}


#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta=theta){

  # see Li, J., Nott, D. J., Fan, Y., & Sisson, S. A. (2017). Extending approximate Bayesian computation methods to high dimensions via a Gaussian copula model. Computational Statistics & Data Analysis, 106, 77-89.
  b=0.1
  
  if(length(theta)==2){
    addterm = 0  
     }
  else{
    addterm = sum(theta[3:length(theta)]^2 /2)
    }
  
  jointprior = exp(-theta[1]^2/200 - (theta[2]-b*theta[1]^2+100*b)^2 / 2 - addterm) 
  
  return(jointprior)
}

sample_prior = function(){
  # see Li, J., Nott, D. J., Fan, Y., & Sisson, S. A. (2017). Extending approximate Bayesian computation methods to high dimensions via a Gaussian copula model. Computational Statistics & Data Analysis, 106, 77-89.
  npar = 20
  diag_cov = diag(npar)
  diag_cov[1,1] = 100
  theta = mvtnorm::rmvnorm(1,mean=rep(0,npar),sigma=diag_cov)
  b = 0.1
  theta[2] = theta[2] + b*theta[1]^2 -100*b
  
  return(as.vector(theta))
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

# :::::::::: Model parameters ::::::::::::::::::::
model_param = NULL

dim_data = 20
num_param = dim_data