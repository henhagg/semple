semple = function(observedData,burnin,K_start,cov_structure,maxiter=100,prior_pdf,sample_prior,jacobian,model,num_iters,factor_cvMH=1.2,
                     mixprob_thresh=0,dim_data,verbose=1,MH_target="likelihood",model_name,model_param=NULL,num_samples_saved,
                     num_simulation_per_iteration,start_time,output_dir)
{

if(num_simulation_per_iteration > num_samples_saved){
  stop("num_simulation_per_iteration > num_samples_saved. Not all simulation in an iteration are saved.
     Consider increasing num_samples_saved or simulations are discarded.")
}

# prior predictive parameters and samples
allpar = replicate(num_priorpred_samples, sample_prior())
sims = apply(allpar, 2, model, model_param=model_param)
write.table(t(allpar), file = paste(output_dir,"/post_sample_iter0.csv", sep = ''), sep = ",", row.names = F, col.names = F)

# receives supplied simulations from the prior predictive
responses = allpar
covariates = sims

# number of parameters in model
num_param = nrow(allpar)

# initialize matrices 
allpar = matrix(data=NA,nrow=num_param,ncol=num_samples_saved) # collects parameters from the prior
sims = matrix(data=NA,nrow=nrow(sims),ncol=num_simulation_per_iteration)
# initialize mixture posterior parameter arrays
post_mix_prob = vector("list", num_iters+1)
covar_proposal = vector("list", num_iters+1)
mean_proposal = vector("list", num_iters+1)


## Set K mixture components in the model
K = K_start
gllim_param = fit_gllim(responses,covariates,K,maxiter,verbose,cov_structure,mixprob_thresh,init_param=NULL)

K = gllim_param$K
A = gllim_param$A
b = gllim_param$b
Sigma = gllim_param$Sigma
c = gllim_param$c
Gamma = gllim_param$Gamma
prior_mix_prob = gllim_param$prior_mix_prob

optpar = gllim_param$optpar

# compute forward relation parameters A_star, b_star and Sigma_star
post_param_list = compute_post_param(A, b, c, Sigma, Gamma, num_param, K, dim_data)
A_star = post_param_list$A_star
b_star = post_param_list$b_star
Sigma_star = post_param_list$Sigma_star

# compute first mixture posterior probabilities (eta)
post_mix_prob[[1]] = compute_post_mix_prob(A, b, c, Sigma, Gamma, prior_mix_prob, K, observedData)

# compute mean and covariance of first mixture proposal
covar_proposal[[1]] = Sigma_star
mean_proposal[[1]]= compute_mean_proposal(A_star, observedData, b_star, num_param, K)

time_vec = rep(NA, num_iters)
for(t in 1:num_iters){
  if(t==1){ # sample from the learned Gaussian mixture
    for(i in 1:num_samples_saved){
      allpar[,i] = par_inflated_mix(probs=post_mix_prob[[t]],means=mean_proposal[[t]],covars=covar_proposal[[t]],factor_cvMH=1)
    }
    write.table(t(allpar), file = paste(output_dir,"/post_sample_iter", t, ".csv", sep = ''), sep = ",", row.names = F, col.names = F)
    
    time_vec[t] = difftime(Sys.time(), start_time, units="secs")
    
    sims = apply(allpar[,1:num_simulation_per_iteration], 2, model, model_param=model_param)
    
    
    responses = allpar[,1:num_simulation_per_iteration]
    covariates = sims
    
  }else{
    # this is so next chain will start at the last accepted parameter
    proposal_old = allpar[,ncol(allpar)]
    
    gllim_param = fit_gllim(responses,covariates,K,maxiter,verbose,cov_structure,mixprob_thresh,init_param = optpar)

    K = gllim_param$K
    A = gllim_param$A
    b = gllim_param$b
    Sigma = gllim_param$Sigma
    c = gllim_param$c
    Gamma = gllim_param$Gamma
    prior_mix_prob = gllim_param$prior_mix_prob

    optpar = gllim_param$optpar
    
    # compute forward relation parameters A_star, b_star and Sigma_star
    post_param_list = compute_post_param(A, b, c, Sigma, Gamma, num_param, K, dim_data)
    A_star = post_param_list$A_star
    b_star = post_param_list$b_star
    Sigma_star = post_param_list$Sigma_star
    
    # compute mixture posterior probabilities (eta)
    post_mix_prob[[t]] = compute_post_mix_prob(A, b, c, Sigma, Gamma, prior_mix_prob, K, observedData)
    
    # propose from learned GLLIM model from the previous iteration
    covar_proposal[[t]] = Sigma_star
    mean_proposal[[t]] = compute_mean_proposal(A_star, observedData, b_star, num_param, K)
    
    #::::::::::::::::: INDEPENDENT PROPOSALS METROPOLIS-HASTINGS (MH) :::::::::::::::::::::::::
    
    # propose from the non-corrected surrogate posterior (q) of the previous iteration
    post_mix_prob_prev = post_mix_prob[[t-1]]
    mean_proposal_prev = mean_proposal[[t-1]]
    covar_proposal_prev = covar_proposal[[t-1]]
    
    if(MH_target=="post"){
      MH_sample_post = sample_MH_surrogatepost(proposal_old, post_mix_prob, mean_proposal, covar_proposal, prior_pdf, t, post_mix_prob_prev,
                                               mean_proposal_prev, covar_proposal_prev, factor_cvMH, num_samples_saved, burnin)
      allparMCMC = MH_sample_post$param
      accrate = MH_sample_post$accrate
    
    }else if(MH_target=="likelihood"){
      MH_sample_lik = sample_MH_surrogatelik(proposal_old, prior_pdf, post_mix_prob_prev, mean_proposal_prev, covar_proposal_prev, factor_cvMH,
                                             A, b, c, Gamma, Sigma, prior_mix_prob, K, observedData, num_samples_saved, burnin)
      allparMCMC = MH_sample_lik$param
      accrate = MH_sample_lik$accrate
      
    }else{
      warning('no valid MH target distribution provided')
    }
    
    write.table(accrate, file = paste(output_dir ,"/accrate_iter", t, ".csv", sep = ''), sep = ",", row.names = F, col.names = F)
    
    allpar = allparMCMC[, seq(1+burnin, num_samples_saved+burnin)] # remove burn in from MC
    write.table(t(allpar), file = paste(output_dir,"/post_sample_iter", t, ".csv", sep = ''), sep = ",", row.names = F, col.names = F)
    
    time_vec[t] = difftime(Sys.time(), start_time, units="secs")
    
    sims = apply(allpar[,1:num_simulation_per_iteration], 2, model, model_param=model_param)
    
    # stack the latest simulations together with previous ones (only done when t>1)
    responses = cbind(responses, allpar[,1:num_simulation_per_iteration])
    covariates = cbind(covariates, sims)
    
  }  # ends the ELSE for t>1
  write.table(time_vec, file = paste(output_dir,"/elapsed_time.csv", sep = ''), sep = ",", row.names = F, col.names = F)
} # end of for-loop



return(list(responses=responses,covariates=covariates,K=K,optpar))
}


# ::::::::::::::::::::: FUNCTION DEFINITIONS ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
fit_gllim = function(responses,covariates,K,maxiter,verbose,cov_structure,mixprob_thresh,init_param){
  # fit prior predictive training data via GLLIM
  mod = gllim(responses,covariates,in_K=K,maxiter=maxiter,in_theta=init_param,verb=verbose,cstr=list(Sigma=cov_structure))
  
  # retrieve MLE parameters for the mixture of f(sims|allpar)  (easier to fit than f(allpar|sims))
  A = mod$A
  b = mod$b
  Sigma = mod$Sigma
  c = mod$c
  Gamma = mod$Gamma
  prior_mix_prob = mod$pi  # these are the prior mixture probabilities denote with \pi_k in Forbes et al
  K = length(prior_mix_prob)
  
  # check if exist components having prior_mix_prob < mixprob_thresh for numerical stability
  id_eliminate = which(prior_mix_prob<mixprob_thresh)
  if(length(id_eliminate)>0){ # if >0 there exist components having small mixture probabilities
    K = K - length(id_eliminate)  # reduce number of components accordingly
    warning('mixprob_thresh reduced the mixture number of components from ', K + length(id_eliminate),  ' to ',K)
    A = A[,,-id_eliminate]
    b = b[,-id_eliminate]
    Sigma = Sigma[,,-id_eliminate]
    c = c[,-id_eliminate]
    Gamma = Gamma[,,-id_eliminate]
    prior_mix_prob = prior_mix_prob[-id_eliminate]
  }
  
  # store the MLEs so we can reuse those as starting values at next optimization round
  optpar=NULL
  optpar$A = A
  optpar$b = b
  optpar$Sigma = Sigma
  optpar$c = c
  optpar$Gamma = Gamma
  optpar$pi = prior_mix_prob
  
  return(list(K=K, A=A, b=b, Sigma=Sigma, c=c, Gamma=Gamma, prior_mix_prob=prior_mix_prob, optpar=optpar))
}

compute_surrogatepost_corrected = function(theta, post_mix_prob, mean_proposal, covar_proposal, prior_pdf, iteration){
  surrogatepost_corrected = mixpdf(theta=theta, probs=post_mix_prob[[1]], means=mean_proposal[[1]], covars=covar_proposal[[1]])
  
  if(iteration >= 2){
    for (i in 2:(iteration)){
      surrogatepost_corrected = mixpdf(theta=theta, probs=post_mix_prob[[i]], means=mean_proposal[[i]], covars=covar_proposal[[i]]) / surrogatepost_corrected
    }
  }
  
  # multiply by prior for even iterations
  # prior factors cancel for odd iterations
  if((iteration %% 2) == 0){
    surrogatepost_corrected = surrogatepost_corrected * prior_pdf(theta)
  }
  
  return(surrogatepost_corrected)
}

# generic function returning the pdf of a generic Gaussian mixture evaluated at some value theta
mixpdf = function(theta,probs,means,covars){
  pdf = 0
  numcomp = length(probs)
  if(numcomp>1){
    for(k in 1:numcomp){
      pdf = pdf + probs[k]*mixtools::dmvnorm(theta,mu=means[,k],sigma=covars[,,k])
    }
  }else{  # make sure that covars is a matrix not an array or dmvnorm will give an error
  #  covars_matrix = matrix(data=NA,nrow=nrow(covars),ncol=ncol(covars))
  #  covars_matrix = covars[,,1]  # the latter is an array, let's strip the third dimension
  #  covars = covars_matrix
    pdf = mixtools::dmvnorm(theta,mu=means,sigma=covars)
  }
  return(pdf)
}

# function returning a sample from a GMM with inflated covariance. 
par_inflated_mix = function(probs, means, covars, factor_cvMH=1){
  numcomp = length(probs)
  
  k = which(rmultinom(1,1,probs)==1)  # samples an index from the multinomial distribution with associated probabilities probs
  if(numcomp>1)
  {
    par_inflated_mix = mixtools::rmvnorm(1,mu=means[,k],sigma=factor_cvMH*covars[,,k]) # sample from k-th component
  }else{
    # make sure that covars is a matrix not an array or dmvnorm will give an error
  #  covars_matrix = matrix(data=NA,nrow=nrow(covars),ncol=ncol(covars))
  #  covars_matrix = covars[,,1]  # the latter is an array, let's strip the third dimension
  #  covars = covars_matrix
    par_inflated_mix = mixtools::rmvnorm(1,mu=means,sigma=factor_cvMH*covars) 
  }
  
}

# compute mixture posterior parameters mean and covariance parameters A_star, b_star and Sigma_star
compute_post_param = function(A, b, c, Sigma, Gamma, num_param, K, dim_data){
  A_star = array(NA,dim=c(num_param,dim_data,K)) # L x d x K
  b_star = array(NA,dim=c(num_param,K))  # L x K
  Sigma_star = array(NA,dim=c(num_param,num_param,K)) # L x L x K
  
  for (k in 1:K){
    if(K==1){
       Sigma_star = solve(solve(Gamma) + t(A)%*%solve(Sigma)%*%A)
       A_star = Sigma_star %*% t(A) %*% solve(Sigma)
       b_star = Sigma_star %*% (solve(Gamma)%*%c - t(A)%*%solve(Sigma)%*%b)
    }
    else
    {
      Sigma_star_k = solve(solve(Gamma[,,k]) + t(A[,,k])%*%solve(Sigma[,,k])%*%A[,,k])
      Sigma_star[,,k] = Sigma_star_k
      A_star[,,k] = Sigma_star_k %*% t(A[,,k]) %*% solve(Sigma[,,k])
      b_star[,k] = Sigma_star_k %*% (solve(Gamma[,,k])%*%c[,k] - t(A[,,k])%*%solve(Sigma[,,k])%*%b[,k])
    }
    
  }
  return(list(A_star=A_star, b_star=b_star, Sigma_star=Sigma_star))
}

# compute mixture posterior probabilities
compute_post_mix_prob = function(A, b, c, Sigma, Gamma, pi, K, observedData){
  post_mix_prob = vector(length=K)
  for (k in 1:K){
    if(K==1){
      c_star = A%*%c + b
      Gamma_star = Sigma + A%*%Gamma%*%t(A)
      post_mix_prob = pi*mixtools::dmvnorm(observedData, mu=c_star, sigma=Gamma_star)
    }
    else{
    c_star_k = A[,,k]%*%c[,k] + b[,k]
    Gamma_star_k = Sigma[,,k] + A[,,k]%*%Gamma[,,k]%*%t(A[,,k])
    post_mix_prob[k] = pi[k]*mixtools::dmvnorm(observedData, mu=c_star_k, sigma=Gamma_star_k)
    }
  }
  post_mix_prob = post_mix_prob/sum(post_mix_prob)
  return(post_mix_prob)
}

# compute mean of mixture proposal distribution
compute_mean_proposal = function(A_star, observedData, b_star, num_param, K){
  mean_proposal = array(NA, dim=c(num_param, K))
  for (k in 1:K){
    if(K==1){
      mean_proposal = A_star%*%observedData + b_star
    }
    else{
    mean_proposal[,k] = A_star[,,k]%*%observedData + b_star[,k]
    }
  }
  return(mean_proposal)
}

# compute mean of surrogate likelihood 
mean_surrogate_likelihood = function(A, b, observedData, theta, K){
  mean_surrogate_likelihood = array(NA, dim=c(length(observedData), K))
  for (k in 1:K){
    if(K==1){
      mean_surrogate_likelihood = A%*%theta + b
    }
    else{
    mean_surrogate_likelihood[,k] = A[,,k]%*%theta + b[,k]
    }
  }
  return(mean_surrogate_likelihood)
}

# compute surrogate likelihood mixture probabilities
likelihood_mix_prob = function(c, Gamma, pi, K, theta){
  likelihood_mix_prob = vector(length=K)
  for (k in 1:K){
    if(K==1){
      likelihood_mix_prob = pi*mixtools::dmvnorm(theta, mu=c, sigma=Gamma)
    }
    else{
    likelihood_mix_prob[k] = pi[k]*mixtools::dmvnorm(theta, mu=c[,k], sigma=Gamma[,,k])
    }
  }
  likelihood_mix_prob = likelihood_mix_prob/sum(likelihood_mix_prob)
  return(likelihood_mix_prob)
}

# sample from corrected surrogate posterior by Metropolis Hastings
sample_MH_surrogatepost = function(proposal_old, post_mix_prob, mean_proposal, covar_proposal, prior_pdf, t, post_mix_prob_prev,
                                   mean_proposal_prev, covar_proposal_prev, factor_cvMH, num_samples_saved, burnin){
  
  allparMCMC = matrix(data=NA,nrow=length(proposal_old),ncol=(num_samples_saved+burnin))
  
  surrogatepost_corrected_old = compute_surrogatepost_corrected(theta=proposal_old, post_mix_prob, mean_proposal, covar_proposal, prior_pdf, iteration=t)
  
  num_accept=0
  for(j in 1:(burnin+num_samples_saved)){
    proposal = par_inflated_mix(probs=post_mix_prob_prev,means=mean_proposal_prev,covars=covar_proposal_prev,factor_cvMH=factor_cvMH)
    
    surrogatepost_corrected = compute_surrogatepost_corrected(theta=proposal, post_mix_prob, mean_proposal, covar_proposal, prior_pdf, iteration=t)
    
    jac = jacobian(theta_old=proposal_old, theta=proposal) # the ratio of transformation jacobians
    
    MHratio = min(1, surrogatepost_corrected/surrogatepost_corrected_old * jac *
                    mixpdf(theta=proposal_old,probs=post_mix_prob_prev,means=mean_proposal_prev,covars=factor_cvMH*covar_proposal_prev)/
                    mixpdf(theta=proposal,probs=post_mix_prob_prev,means=mean_proposal_prev,covars=factor_cvMH*covar_proposal_prev))
    
    if(is.nan(MHratio)){
      allparMCMC[,j] = proposal_old
      
    }else if(runif(1)<MHratio){
      proposal_old = proposal
      allparMCMC[,j] = proposal
      
      surrogatepost_corrected_old = surrogatepost_corrected
      
      if(j > burnin){
        num_accept = num_accept + 1
      }
      
    }
    else{
      allparMCMC[,j] = proposal_old
    }
  }
  
  accrate = num_accept/num_samples_saved
  
  return(list(param=allparMCMC, accrate=accrate))
}

# sample from [surrogate likelihood * prior] by Metropolis Hastings
sample_MH_surrogatelik = function(proposal_old, prior_pdf, post_mix_prob_prev, mean_proposal_prev, covar_proposal_prev, factor_cvMH,
                                  A, b, c, Gamma, Sigma, prior_mix_prob, K, observedData, num_samples_saved, burnin){
  
  allparMCMC = matrix(data=NA,nrow=length(proposal_old),ncol=(burnin+num_samples_saved))
  
  likelihood_mix_prob_old = likelihood_mix_prob(c, Gamma, prior_mix_prob, K, proposal_old)
  mean_surrogate_likelihood_old = mean_surrogate_likelihood(A, b, observedData, proposal_old, K)
  surrogatelik_old = mixpdf(theta=observedData, probs=likelihood_mix_prob_old, mean=mean_surrogate_likelihood_old, covars=Sigma)
  
  num_accept=0
  for(j in 1:(burnin+num_samples_saved)){
    proposal = par_inflated_mix(probs=post_mix_prob_prev,means=mean_proposal_prev,covars=covar_proposal_prev,factor_cvMH=factor_cvMH)
    
    jac = jacobian(theta_old=proposal_old,theta=proposal) # the ratio of transformation jacobians
    
    likelihood_mix_prob = likelihood_mix_prob(c, Gamma, prior_mix_prob, K, proposal)
    mean_surrogate_likelihood = mean_surrogate_likelihood(A, b, observedData, t(proposal), K)
    surrogatelik = mixpdf(theta=observedData, probs=likelihood_mix_prob, mean=mean_surrogate_likelihood, covars=Sigma)
    
    MHratio = min(1,surrogatelik/surrogatelik_old * jac * prior_pdf(proposal)/prior_pdf(proposal_old) *
                    mixpdf(theta=proposal_old,probs=post_mix_prob_prev,means=mean_proposal_prev,covars=factor_cvMH*covar_proposal_prev)/
                    mixpdf(theta=proposal,probs=post_mix_prob_prev,means=mean_proposal_prev,covars=factor_cvMH*covar_proposal_prev))
    
    if(is.nan(MHratio)){
      allparMCMC[,j] = proposal_old
    }else if(runif(1)<MHratio){
      proposal_old = proposal
      allparMCMC[,j] = proposal
      
      surrogatelik_old = surrogatelik
      
      if(j > burnin){
        num_accept = num_accept + 1
      }
    }
    else{
      allparMCMC[,j] = proposal_old
    }
  }
  
  accrate = num_accept/num_samples_saved
  
  return(list(param=allparMCMC, accrate=accrate))
}
