# ::::::::::::::::::::::::::: Model parameters :::::::::::::::::::::::::::::::::
model_param = list(T = 100)
dim_data = 10
num_param = 10

#::::::::::::::::::::::::::::: Generative model ::::::::::::::::::::::::::::::::
design_matrix = as.matrix(read.csv(file = "models/bernoulli_glm/design_matrix.csv", header = FALSE))
stimulus_I = as.vector(as.matrix(
  read.csv(file = "models/bernoulli_glm/stimulus_I.csv", header = FALSE)
))

model = function(param, model_param = NULL) {
  psi = design_matrix %*% param
  z = 1 / (1 + exp(-psi))
  y = as.numeric(runif(dim(design_matrix)[1]) < z)
  
  num_spikes = sum(y)
  sta = convolve(c(y, rep(0, 8)), stimulus_I, type = "filter")
  
  return(c(num_spikes, sta))
}

gaussian_cov = function(num_param = 10) {
  M = num_param - 1
  D = diag(M)
  diag(D[-1, ]) = -1
  F = D %*% D + diag(seq(0, M - 1) / M) ^ 0.5
  
  B = matrix(0, nrow = M + 1, ncol = M + 1)
  B[1, 1] = 2
  B[2:(M + 1), 2:(M + 1)] = solve(t(F) %*% F)
  
  return(B)
}

#:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::
prior_pdf = function(theta = theta) {
  cov_mat = gaussian_cov(num_param = length(theta))
  return(mvtnorm::dmvnorm(theta, mean = rep(0, length(theta)), sigma = cov_mat))
}

sample_prior = function() {
  cov_mat = gaussian_cov(num_param = num_param)
  return(as.vector(mvtnorm::rmvnorm(
    1, mean = rep(0, num_param), sigma = cov_mat
  )))
}

#::::::::: TRANSFORMATION JACOBIAN ::::::::::::::::::::::
jacobian = function(theta_old, theta) {
  jacobian = 1 # no transformation jacobian needed here
  return(jacobian)
}