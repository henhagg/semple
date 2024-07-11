library(mvtnorm)

#:::::::::::::: Model parameters :::::::::::::::::::::::::::
# Microphones configuration and Student parameters
micr1=c(-0.5,0)
micr2=c(0.5,0)
micr1p=c(0,-0.5)
micr2p=c(0,0.5)
# Student Noise 
sigmaT=0.01
dofT=3
ny=10

model_param = list(micr1=micr1, micr2=micr2, micr1p=micr1p, micr2p=micr2p, sigmaT=sigmaT, dofT=dofT, ny=ny)
true_location = c(1.5,1)
dim_data = ny
num_param = 2

#::::::::::::::::::::::::: Generative model ::::::::::::::::::::::::::::::::::::
# Functions taken from https://github.com/Trung-TinNGUYEN/GLLiM-ABC
simuITDT=function(x12,m1,m2,sigmaT,dofT,ny){
  micr11=m1[1]
  micr12=m1[2]
  micr21=m2[1]
  micr22=m2[2]
  x1=x12[1]
  x2=x12[2]
  d1=sqrt((x1-micr11)^2 + (x2-micr12)^2)
  d2=sqrt((x1-micr21)^2 + (x2-micr22)^2)
  Mutest = rep(abs(d1-d2),ny)
  Sigmatest=diag(sigmaT,ny)
  #ysimu= Mutest+ rmvt(1, sigma=Sigmatest, df=1)
  ysimu= mvtnorm::rmvt(1, delta=Mutest, sigma=Sigmatest, df=dofT)
  ysimu
}

simuITDTmix=function(x12,m1,m2,m1p,m2p,sigmaT,dofT,ny){
  usimu=runif(1,0,1)
  if(runif(1,0,1)>0.5)
    ysimu=simuITDT(x12,m1,m2,sigmaT,dofT,ny)
  else ysimu=simuITDT(x12,m1p,m2p,sigmaT,dofT,ny)
  ysimu
}

model = function(param, model_param=NULL){
  list2env(model_param, environment())
  
  return(simuITDTmix(param, m1=micr1,m2=micr2,m1p=micr1p,m2p=micr2p,sigmaT=sigmaT,dofT=dofT,ny=ny))
}

#:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta=theta){
  jointprior = dunif(theta[1],-2,2)*dunif(theta[2],-2,2)
  return(jointprior)
}

sample_prior = function(){
  prior_sample = runif(n=2, min=-2, max=2)
  return(prior_sample)
}

#:::::::::::::::::::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::::::::
jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

#::::::::::::::: METROPOLIS-HASTINGS HELP FUNCTIONS ::::::::::::::::::::::::::::
# Functions taken from https://github.com/Trung-TinNGUYEN/GLLiM-ABC
UnpostITDT=function(yobs, x12, m1, m2, sigmaT, dofT){
  x1=x12[1]
  x2=x12[2]
  micr11=m1[1]
  micr12=m1[2]
  micr21=m2[1]
  micr22=m2[2]
  ny=length(yobs)
  
  d1=sqrt((x1-micr11)^2 + (x2-micr12)^2)
  d2=sqrt((x1-micr21)^2 + (x2-micr22)^2)
  Mut = rep(abs(d1-d2), ny)
  Sigmat=diag(sigmaT,ny)
  return(mvtnorm::dmvt(yobs, delta=Mut, sigma=Sigmat, df=dofT, log=F))
}

logunpostITDTmix=function(yobs, x12, m1, m2, m1p, m2p, sigmaT, dofT){
  x1=x12[1]
  x2=x12[2]
  if ((x1< -2) | (x1>2) |(x2< -2) | (x2>2) )
    # attention delta = mode not mean
    return(-Inf) else return(log(0.5*UnpostITDT(yobs, x12, m1, m2, sigmaT, dofT) + 0.5*UnpostITDT(yobs, x12, m1p, m2p, sigmaT, dofT)))
}

LikeITDTunMix=function(theta,yobs,sigmaT,dofT, m1, m2, mp1, mp2){
  ny = length(yobs)
  itd = abs(sqrt(sum((theta-m1)^2))-sqrt(sum((theta-m2)^2)))
  c1=(1+1/(dofT*sigmaT)*sum((yobs-itd)^2))^(-(dofT+ny)/2)
  itdp = abs(sqrt(sum((theta-mp1)^2))-sqrt(sum((theta-mp2)^2)))
  0.5*c1 + 0.5*((1+1/(dofT*sigmaT)*sum((yobs-itdp)^2))^(-(dofT+ny)/2))
}

PostPdfITDMix=function(N_grid, y){
  # uniform prior on [-2,2]^2
  x1 = seq(-2,2, length = N_grid)
  x2 = seq(-2,2, length = N_grid)
  grid = expand.grid(x1 = x1, x2 = x2)
  z = apply(grid,1, LikeITDTunMix, yobs=y, sigmaT=sigmaT,dofT=dofT, m1=micr1, m2= micr2, mp1=micr1p, mp2=micr2p)
  
  # Full posterior (L=2) unormalized, ok for contour plots
  full_df = cbind(grid, z)
  # ITD_df
  list("postdf"= full_df)
}