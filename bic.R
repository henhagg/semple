library(xLLiM)

compute_bic = function(model_name,
                       num_priorpred_samples,
                       K_values,
                       cov_structure,
                       gllim_verb = 0,
                       save_result_to_csv = FALSE,
                       save_plot = FALSE) {
  source(paste("models/", model_name, "/", model_name, "_model.R", sep = ""))
  
  D = dim_data
  L = num_param
  
  set.seed(1) # for reproducibility
  prior_param = replicate(num_priorpred_samples, sample_prior())
  prior_sims = apply(prior_param, 2, model, model_param = model_param)
  
  bic_values = rep(NA, length(K_values))
  
  for (K in K_values) {
    print(K)
    mod = gllim(
      prior_param,
      prior_sims,
      in_K = K,
      maxiter = 300,
      verb = gllim_verb,
      cstr = list(Sigma = cov_structure)
    )
    bic_values[which(K_values == K)] = mod$nbpar * log(num_priorpred_samples) - 2 *
      mod$LLf
  }
  
  if (save_result_to_csv) {
    save_bic_to_csv(K_values,
                    bic_values,
                    model_name,
                    num_priorpred_samples,
                    cov_structure)
  }
  
  if (save_plot) {
    save_bic_plot(K_values,
                  bic_values,
                  model_name,
                  num_priorpred_samples,
                  cov_structure)
  }
  
  return(bic_values)
}

save_bic_to_csv = function(K_values,
                           bic_values,
                           model_name,
                           num_priorpred_samples,
                           cov_structure) {
  save_dir = file.path(getwd(), "results", model_name, "semple", "bic")
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  write.table(
    rbind(K_values, bic_values),
    file = paste(
      save_dir,
      "/bic_",
      model_name,
      "_priorpred",
      num_priorpred_samples,
      "_cov",
      cov_structure,
      "_Kmin",
      min(K_values),
      "_Kmax",
      max(K_values),
      ".csv",
      sep = ''
    ),
    sep = ",",
    row.names = F,
    col.names = F
  )
}

save_bic_plot = function(K_values,
                         bic_values,
                         model_name,
                         num_priorpred_samples,
                         cov_structure) {
  save_dir = file.path(getwd(), "figures", model_name, "bic")
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  pdf(
    file = paste(
      save_dir,
      "/bic_",
      model_name,
      "_priorpred",
      num_priorpred_samples,
      "_cov",
      cov_structure,
      "_Kmin",
      min(K_values),
      "_Kmax",
      max(K_values),
      ".pdf",
      sep = ''
    )
  )
  plot(
    K_values,
    bic_values,
    type = "b",
    xlab = "K",
    ylab = "BIC"
  )
  dev.off()
}

plot_bic_from_file = function(model_name,
                              num_priorpred_samples,
                              cov_structure,
                              K_min,
                              K_max) {
  bic_table = read.csv(
    file = paste(
      "results/",
      model_name,
      "/semple/bic/bic_",
      model_name,
      "_priorpred",
      num_priorpred_samples,
      "_cov",
      cov_structure,
      "_Kmin",
      K_min,
      "_Kmax",
      K_max,
      ".csv",
      sep = ""
    ),
    header = FALSE
  )
  print(bic_table)
  plot(unlist(bic_table[1, ]), unlist(bic_table[2, ]), type = 'b')
}


#::::::::::::::::::::::::::::::: TWO MOONS :::::::::::::::::::::::::::::::::::::
K_values = seq(10, 70, 10)
bic_values = compute_bic(
  model_name = "two_moons",
  num_priorpred_samples = 2500,
  K_values = K_values,
  cov_structure = "",
  gllim_verb = 0,
  save_result_to_csv = TRUE,
  save_plot = TRUE
)
plot(K_values,
     bic_values,
     type = "b",
     xlab = "K",
     ylab = "BIC")

#::::::::::::::::::::::::::: MULTIPLE HYPERBOLOID ::::::::::::::::::::::::::::::
K_values = seq(10, 50, 5)
bic_values = compute_bic(
  model_name = "hyperboloid",
  num_priorpred_samples = 10000,
  K_values = K_values,
  cov_structure = "i",
  gllim_verb = 0,
  save_result_to_csv = TRUE,
  save_plot = TRUE
)
plot(K_values,
     bic_values,
     type = "b",
     xlab = "K",
     ylab = "BIC")