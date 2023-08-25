# Sequential Mixture Posterior and Likelihood Estimation (SeMPLE)

## Setup
Clone the SBIBM fork https://github.com/henhagg/sbibm into the parent directory of the semple repository to get local access to the observed data sets and reference samples of the Two Moons and SLCP models.

Run the following command in R to get the correct version of xLLiM: 
```commandline
devtools::install_github("epertham/xLLiM", ref = "devel")
```

## Running SeMPLE
To run SeMPLE with with a specific model/task run the corresponding "model_name.R" script, e.g. "two_moons.R". The algorithm outputs are saved into the results directory.

Each model can be found in models/model_name/model_name_model.R. A function to simulate data from the model as well as the prior is defined here. The hyperboloid model and the Ornstein-Uhlenbeck process both have a single observed dataset found in models/model_name/num_observation_1. The Two Moons model and the the SLCP model instead uses the observed datasets from SBIBM by fetching them locally from the SBIBM repository cloned into the parent directory of the semple repository.

## SBIBM
To run SNPE-C or SNL to produce results to be compared to SeMPLE results, install the SBIBM fork https://github.com/henhagg/sbibm. It contains modifications to output posterior samples from each SNPE-C and SNL algorithm iteration instead of only from the last algorithm iteration. Additionally, it is modified to output the runtime of SNPE-C and SNL. Furthermore, the fork contains implementations of the hyperboloid model and Orstein-Uhlenbeck process.

## Files explanation
model_name.R - runs SeMPLE with a specific model.

gaumixfit_sbibm.R - contains implementation of the SeMPLE algorithm in the form of the gaumixfit function.

compute_metrics.py - is used to compute performance metrics for a subfolder in the results directory, the metric results are saved in the same subfoler.

plot_results.py - contains functions to produce and save plots of results from results/ into figures/.

run_sbibm.py - is used to run the SNPE-C and SNL algorithms using SBIBM.

bic.R - is used to compute the BIC of the number of mixture components used in SeMPLE.