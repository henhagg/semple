library(latex2exp)  # for LaTeX labels

plot_all_param_hist = function(file_path, true_par, prior_mean, output_file=NULL, plot_width=7, plot_height=5){
  posterior_samples = read.csv(file_path, header=FALSE)
  
  if(!is.null(output_file)){
    pdf(output_file, width=plot_width, height=plot_height)
  }
  
  par(mfrow=c(2,4), mai = c(1, 0.2, 0.2, 0.2))
  
  for(par_idx in 1:8){
    hist(posterior_samples[,par_idx],
         freq=F,
         yaxt='n',
         xlim=c(xlim_lower[par_idx],xlim_upper[par_idx]),
         xlab = TeX(param_names[par_idx]),
         main=NULL,
         ylab=NULL)
    x<-seq(xlim_lower[par_idx], xlim_upper[par_idx], by=0.02)
    curve(dnorm(x, prior_mean[par_idx], sqrt(0.5)), add=TRUE) # prior
    abline(v=true_par[par_idx], lwd=2, col="red")
  }
  
  if(!is.null(output_file)){
    dev.off()
  }
  
}

plot_all_param_kde = function(file_path, true_par, prior_mean, output_dir=NULL, plot_width=10, plot_height_5){
  posterior_samples = read.csv(file_path, header=FALSE)
  
  if(!is.null(output_file)){
    pdf(output_file, width=plot_width, height=plot_height)
  }
  
  par(mfrow=c(2,4), mai = c(1, 0.2, 0.2, 0.2))
  
  for(par_idx in 1:8){
    kde = density(posterior_samples[,par_idx])
    plot(x=kde$x,
         y=kde$y,
         type="l",
         yaxt='n',
         xlim=c(xlim_lower[par_idx],xlim_upper[par_idx]),
         xlab = TeX(param_names[par_idx]),
         main=NULL,
         ylab=NULL)
    x<-seq(xlim_lower[par_idx], xlim_upper[par_idx], by=0.02)
    curve(dnorm(x, prior_mean[par_idx], sqrt(0.5)), add=TRUE) # prior
    abline(v=true_par[par_idx], lwd=2, col="red")
  }
  
  if(!is.null(output_file)){
    dev.off()
  }
}

plot_multi_alg_kde = function(file_path_list, alg_names, plot_colors, true_par,
                              prior_mean, output_file=NULL, plot_width=10, plot_height=5){
  # load all csv files
  num_algs = length(file_path_list)
  post_sample_list = list()
  for(a in 1:num_algs){
    post_sample_list[[a]] = read.csv(file_path_list[a], header=FALSE)
  }
  
  # compute ylim before plotting
  max_kde_val = rep(0,8)
  for(par_idx in 1:8){
    for(alg_idx in 1:num_algs){
      posterior_samples = post_sample_list[[alg_idx]]
      kde = density(posterior_samples[,par_idx])
      max_kde_val[par_idx] = max(max_kde_val[par_idx], max(kde$y))
    }
  }
  
  if(!is.null(output_file)){
    pdf(output_file, width=plot_width, height=plot_height)
  }
  
  par(mfrow=c(2,4), mai = c(0.7, 0.2, 0.2, 0.2))
  
  for(par_idx in 1:8){
    posterior_samples = post_sample_list[[1]]
    kde = density(posterior_samples[,par_idx])
    plot(x=kde$x,
         y=kde$y,
         type="l",
         col=plot_colors[1],
         yaxt='n',
         xlim=c(xlim_lower[par_idx],xlim_upper[par_idx]),
         xlab=TeX(param_names[par_idx]),
         ylim=c(0,max_kde_val[par_idx]),
         main=NULL,
         ylab=NULL,
         cex.lab=1.2,
         cex.axis=1.1)
    for(alg_idx in 2:num_algs){
      posterior_samples = post_sample_list[[alg_idx]]
      kde = density(posterior_samples[,par_idx])
      lines(kde, col=plot_colors[alg_idx])
    }
    
    x<-seq(xlim_lower[par_idx], xlim_upper[par_idx], by=0.02)
    curve(dnorm(x, prior_mean[par_idx], sqrt(0.5)), add=TRUE) # prior
    abline(v=true_par[par_idx], lwd=2, col="red")
  }
  
  # add legend
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = 'l', bty = 'n', xaxt = 'n', yaxt = 'n')
  legend("center", legend=alg_names, col=plot_colors, lwd = 2, xpd = TRUE, horiz = TRUE, cex = 1.1, seg.len=1, bty = 'n')
  
  if(!is.null(output_file)){
    dev.off()
  }
}

param_names = c(r'($\log \, \delta$)', r'($\log \, \gamma$)', r'($\log \, k$)',
                r'($\log \, m_0$)', r'($\log \, scale$)', r'($\log \, t_0$)',
                r'($\log \, offset$)', r'($\log \, \sigma$)')

true_par_obs1 = c(-0.694, -3, 0.027, 5.704, 0.751, -0.164, 2.079, -2) # observation 1
true_par_obs2 = c(-1.827, -3.138, -0.426, 6.869, 0.697, 0.465, 2.787, -1.993) # observation 2
true_par_obs3 = c(-1.153, -3.501, -0.181, 4.601, 0.386, -2.121, 2.185, -2.578) # observation 3
true_par_obs4 = c(-1.297, -2.134, 1.118, 5.577, 0.938, 0.12, 1.122, -2.854) # observation 4
true_par_obs5 = c(-0.243, -4.596, 0.168, 5.573, -0.086, 1.234, 1.392, -1.655) # observation 5

true_par_matrix = matrix(c(true_par_obs1,
                           true_par_obs2,
                           true_par_obs3,
                           true_par_obs4,
                           true_par_obs5),
                         ncol=5)

prior_mean = c(-0.694, -3, 0.027, 5.704, 0.751, -0.164, 2.079, -2)

xlim_upper = c(1.5, -1, 2.5, 8, 3, 2, 4, 0.5)
xlim_lower = c(-2.5, -5, -2.5, 3,-1.5, -2, 0, -4.5)


############## HISTOGRAM
# plot_all_param_hist(file_path = "results/mrna/snpe/30k_10rounds/obs1/post_sample_iter10.csv",
#                     true_par = true_par_obs1,
#                     prior_mean = prior_mean)
# 
plot_all_param_hist(file_path = "results/mrna/semple/30k_3rounds/obs1/post_sample_iter3.csv",
                    true_par = true_par_obs1,
                    prior_mean = prior_mean,
                    output_file = "results/mrna/mRNA_semple.pdf")

############# KDE ALL ALGORITHMS
# for(obs_ind in 1:5){
#   plot_multi_alg_kde(file_path_list = c(paste("results/mrna/snpe/30k_10rounds/obs",obs_ind,"/post_sample_iter10.csv", sep=""),
#                                         paste("results/mrna/snle/30k_10rounds/obs",obs_ind,"/post_sample_iter10.csv", sep=""),
#                                         paste("results/mrna/semple/30k_3rounds/obs",obs_ind,"/post_sample_iter3.csv", sep="")),
#                      alg_names = c("SNPE-C", "SNL", "SeMPLE"),
#                      plot_colors = c("magenta", "blue", "darkgreen"),
#                      true_par = true_par_matrix[,obs_ind],
#                      prior_mean = prior_mean,
#                      output_file=paste("results/mrna/mrna_all_alg_obs",obs_ind,".pdf", sep=""),
#                      plot_width=7,
#                      plot_height=5)
# }



############ SEMPLE ALL ROUNDS
# for(obs_ind in 1:5){
#   plot_multi_alg_kde(file_path_list = c(paste("results/mrna/semple/30k_3rounds/obs",obs_ind,"/post_sample_iter1.csv",sep=""),
#                                         paste("results/mrna/semple/30k_3rounds/obs",obs_ind,"/post_sample_iter2.csv",sep=""),
#                                         paste("results/mrna/semple/30k_3rounds/obs",obs_ind,"/post_sample_iter3.csv",sep="")),
#                      alg_names = c("SeMPLE round 1", "SeMPLE rounds 2", "SeMPLE round 3"),
#                      plot_colors = c("magenta", "blue", "darkgreen"),
#                      true_par=true_par_matrix[,obs_ind],
#                      prior_mean=prior_mean,
#                      output_file=paste("results/mrna/semple_allrounds_obs",obs_ind,".pdf", sep=""),
#                      plot_width=7,
#                      plot_height=5)
# }







