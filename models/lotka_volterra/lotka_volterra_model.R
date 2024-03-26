# :::::::::: Model parameters ::::::::::::::::::::
model_param = NULL

dim_data = 9
num_param = 4

theta_min = -6
theta_max = 2
sigma_min = log(0.5)
sigma_max = log(50)
#:::: Generative model ::::::::::::::::::::::::::::::::

means_trim = as.vector(unlist(read.csv(file="models/lotka_volterra/means_trim00125_N10000_sd10.csv", header = FALSE)))
sds_trim = as.vector(unlist(read.csv(file="models/lotka_volterra/sds_trim00125_N10000_sd10.csv", header = FALSE)))

ssinit <- function(vec)
{
  ac23=as.vector(acf(vec,lag.max=2,plot=FALSE)$acf)[2:3]
  c(mean(vec),log(var(vec)+1),ac23)  # notice we take log(var+1) not to risk taking the log of zero
}
ssi <- function(ts)
{
  c(ssinit(ts[,1]),ssinit(ts[,2]),cor(ts[,1],ts[,2]))
}

model<-function(param, model_param=NULL){
  th = exp(param)
  trajectory = simTs(c(x1=50,x2=100),0,30,0.2,stepLVc,c(th1=th[1],th2=th[2],th3=th[3]))
  trajectory_noise = trajectory + matrix( rnorm(150*2,mean=0,sd=th[4]), 150, 2)
  standardized_summary = (ssi(trajectory_noise)-means_trim) / sds_trim
  return(standardized_summary)
}
#::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta){
  dunif(theta[1],theta_min,theta_max)*dunif(theta[2],theta_min,theta_max)*dunif(theta[3],theta_min,theta_max)*
    dunif(theta[4],sigma_min,sigma_max)
}

sample_prior = function(){
  c(runif(3, min=theta_min, max=theta_max), runif(1, min=sigma_min, max=sigma_max))
}
#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::

jacobian = function(theta_old,theta){
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}

############################ Generate new observation ##############################
# sd_noise = 30
# trajectory = simTs(c(x1=50,x2=100),0,30,0.2,stepLVc,c(th1=1,th2=0.005,th3=0.6))
# plot(trajectory)
# trajectory_noisy = trajectory + matrix( rnorm(150*2,mean=0,sd=sd_noise), 150, 2)
# plot(trajectory_noisy)
# write.table(trajectory_noisy, "models/lotka_volterra/num_observation_1/observation_sd30.csv", sep = ",", row.names = F, col.names = F)

########################### Compute mean/sds of summary statistics #############
# library(chemometrics) # only useful to compute trimmed standard deviations (to standardize the ABCsummaries)
# library(truncnorm)
# set.seed(1)
# N = 10000
# prior=cbind(th1=runif(N,theta_min,theta_max),th2=runif(N,theta_min,theta_max),th3=runif(N,theta_min,theta_max))
# rows=lapply(1:N,function(i){exp(prior[i,])})
# samples=mclapply(rows,function(th){simTs(c(x1=50,x2=100),0,30,0.2,stepLVc,th) + matrix(rnorm(150*2,mean=0,sd=sd_noise), 150, 2)})
# sumstats=mclapply(samples,ssi) # computes the 9 summaries for each simulated dataset
# 
# # computes trimmed means and trimmed standard deviations for the summary stats
# sds_trim=apply(sapply(sumstats,c),1,sd_trim,trim=0.0125)  # compute trimmed SD by disregarding NaNs cases
# means_trim=apply(sapply(sumstats,c),1,mean,na.rm=T,trim=0.0125) # trim on cases without NaN
# median = apply(sapply(sumstats,c),1,median)
# 
# # save to file
# write.table(t(sds_trim), "models/lotka_volterra/sds_trim00125_N10000_sd10.csv", sep = ",", row.names = F, col.names = F)
# write.table(t(means_trim), "models/lotka_volterra/means_trim00125_N10000_sd10.csv", sep = ",", row.names = F, col.names = F)

