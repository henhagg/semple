library(Rcpp)

sourceCpp("models/mrna/mrna_model.cpp") # source Rcpp implementation

# :::::::::: Model parameters ::::::::::::::::::::
model_param = NULL

num_param = 8
dim_data = 60

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::
jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
#               delta,  gamma,    k,    m_0, scale,    t_0, offset, sigma
prior_mean = c( -0.694,    -3, 0.027, 5.704, 0.751, -0.164,  2.079, -2)
prior_sd = rep(sqrt(0.5), 8)

prior_pdf = function(param){
  dnorm(param[1], mean=prior_mean[1], sd=prior_sd[1])*
  dnorm(param[2], mean=prior_mean[2], sd=prior_sd[2])*
  dnorm(param[3], mean=prior_mean[3], sd=prior_sd[3])*
  dnorm(param[4], mean=prior_mean[4], sd=prior_sd[4])*
  dnorm(param[5], mean=prior_mean[5], sd=prior_sd[5])*
  dnorm(param[6], mean=prior_mean[6], sd=prior_sd[6])*
  dnorm(param[7], mean=prior_mean[7], sd=prior_sd[7])*
  dnorm(param[8], mean=prior_mean[8], sd=prior_sd[8])
}

sample_prior = function(){
  c(rnorm(1, mean=prior_mean[1], sd=prior_sd[1]),
    rnorm(1, mean=prior_mean[2], sd=prior_sd[2]),
    rnorm(1, mean=prior_mean[3], sd=prior_sd[3]),
    rnorm(1, mean=prior_mean[4], sd=prior_sd[4]),
    rnorm(1, mean=prior_mean[5], sd=prior_sd[5]),
    rnorm(1, mean=prior_mean[6], sd=prior_sd[6]),
    rnorm(1, mean=prior_mean[7], sd=prior_sd[7]),
    rnorm(1, mean=prior_mean[8], sd=prior_sd[8]))
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

# ::::::::::::: R implementation of model :::::::::::::::::::::::::::
# gfp_drift = function(delta, gamma, k, m, p){
#   c(-delta*m, k*m-gamma*p)
# }
# 
# gfp_diffusion = function(delta, gamma, k, m ,p){
#   c(delta*m, k*m+gamma*p)
# }
# 
# measurement = function(p, scale, offset){
#   log(scale*p + offset)
# }
# 
# euler_maruyama = function(delta, gamma, k, m0, t0, tt, dt){
#   if (t0 > 30){
#     return(matrix(c(m0,0), nrow=1, ncol=2))
#   }
# 
#   n_points = ceiling(((tt+dt)-t0)/dt)
#   xs = matrix(NA, nrow=n_points, ncol=2)
#   xs[1,] = c(m0,0)
# 
#   dw = rnorm(2*(n_points-1), mean=0, sd=sqrt(dt))
# 
#   for (i in 2:n_points) {
#     x = xs[i-1,]
# 
#     drift = gfp_drift(delta, gamma, k, m=x[1], p=x[2])
#     diffusion = gfp_diffusion(delta, gamma, k, m=x[1], p=x[2])
#     # drift = c(-delta*x[1], k*x[1]-gamma*x[2])
#     # diffusion = c(delta*x[1], k*x[1]+gamma*x[2])
# 
#     xtemp = x + drift*dt + sqrt(diffusion) * dw[(2*i-3):((2*i)-2)] # rnorm(1,mean=0,sd=sqrt(dt))
#     xtemp[xtemp<0] = 0
#     xs[i,] = xtemp
#   }
#   return(xs)
# }
# 
# model = function(logparam){
#   param = exp(logparam)
#   delta = param[1]
#   gamma = param[2]
#   k = param[3]
#   m0 = param[4]
#   scale = param[5]
#   t0 = param[6]
#   offset = param[7]
#   sigma = param[8]
# 
#   tt = 30
#   dt = 0.01
#   if(t0 > tt){
#     ts = t0
#   }else{
#     ts = seq(t0,tt+dt,dt)
#   }
# 
#   sol_euler = euler_maruyama(delta, gamma, k, m0=m0, t0=t0, tt=tt, dt=dt)
#   y = measurement(sol_euler[,2], scale, offset)
#   t_out = seq(tt/dim_data, tt, tt/dim_data)
#   sol = approx(x=ts, y=y, xout=t_out, yleft=log(offset), yright=y[length(y)], method="linear")$y
#   sol_with_noise = sol + rnorm(dim_data, 0, sigma)
# 
#   return(sol_with_noise)
# }
