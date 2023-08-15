#:::: Generative model ::::::::::::::::::::::::::::::::

model = function(param, model_param=NULL){
  a = runif(1, min=-pi/2,max =pi/2)
  r = rnorm(1,0.1,0.01)
  p = c(r*cos(a)+0.25,r*sin(a))
  simulatedData = p + c(-abs(param[1]+param[2])/sqrt(2), (-param[1]+param[2])/sqrt(2)  )
  return(simulatedData)
}


#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta=theta){
  jointprior = dunif(theta[1],-1,1)*dunif(theta[2],-1,1)
  return(jointprior)
}

sample_prior = function(){
  theta = runif(2, min=-1, max=1)
  return(theta)
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

# :::::::::: Model parameters ::::::::::::::::::::
model_param = NULL

dim_data = 2
num_param = 2

