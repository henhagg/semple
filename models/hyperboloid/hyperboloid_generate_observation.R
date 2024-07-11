library(ggplot2)
source("models/hyperboloid/hyperboloid_model.R")

#:::::::::::::::::::::::::: GENERATE OBSERVATION::::::::::::::::::::::::::::::::
set.seed(1)
ytargetITD = simuITDTmix(true_location, micr1, micr2, micr1p, micr2p, sigmaT, dofT, ny = dim_data)

# plot true posterior contours
dfconfig = data.frame(matrix(c(-0.5, 0, 0.5, 0, 0, -0.5, 0, 0.5, 1.5, 1), 5, 2, byrow = T))
N_grid = 500
ITD_df = PostPdfITDMix(N_grid, unlist(ytargetITD))$postdf
v = ggplot(ITD_df) +  xlab("x") + ylab("y") + theme(aspect.ratio = 1) + xlim(-2, 2) + ylim(-2, 2)
v + geom_contour(aes(x1, x2, z = z), color = "blue", bins = 7) + geom_abline(
  slope = 0 ,
  intercept = 0,
  linetype = "dashed",
  linewidth = .3
) + geom_point(
  data = dfconfig,
  mapping = aes(x = X1, y = X2),
  size = 2.5,
  color = "black"
)

# write observation to file
write.table(
  ytargetITD,
  file = "models/hyperboloid/num_observation_2/observation.csv",
  sep = ",",
  row.names = F,
  col.names = F
)