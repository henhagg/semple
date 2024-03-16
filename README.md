# Sequential Mixture Posterior and Likelihood Estimation (SeMPLE)

SeMPLE is a framework for simulation-based inference that learns surrogate models for the likelihood function and the posterior distribution of model parameters, and outputs draws from the posterior. It provides a faster and lightweight solution compared to samplers using neural networks for density estimation, while retaining accuracy.

## Setup
SeMPLE is written in R. To run SeMPLE you need to install the xLLiM R package that gives access to the GLLiM function. Follow the instructions below, do NOT use the version provided on CRAN.

Run the following command in R to get the correct version of xLLiM (requires the R package "devtools" to be installed): 
```commandline
devtools::install_github("epertham/xLLiM", ref = "master")
```
Some of the tasks use data and reference posterior draws provided by the SBIBM Python package. Clone the SBIBM fork https://github.com/henhagg/sbibm into the parent directory of the semple repository to get local access to the observed data sets and reference posterior samples.



## Running SeMPLE
To run SeMPLE with with a specific model/task run the corresponding "model_name.R" script, e.g. "two_moons.R". The algorithm outputs are saved into the results directory.

Model definitions can be found in models/model_name/model_name_model.R. A function to simulate data from the model as well as the prior is defined there. The hyperboloid model, the Ornstein-Uhlenbeck and the twisted-prior models have a single observed dataset found in models/model_name/num_observation_1. The Two Moons, Bernoulli GLM and the the SLCP models instead use the observed datasets from SBIBM, by fetching them locally from the SBIBM repository cloned into the parent directory of the "semple" repository.

## SBIBM
To run SNPE-C or SNL to produce results to be compared to SeMPLE results, install the SBIBM fork https://github.com/henhagg/sbibm. It contains modifications to output posterior samples from each SNPE-C and SNL algorithm iteration instead of only from the last algorithm iteration. Additionally, it is modified to output the runtime of SNPE-C and SNL. Furthermore, the fork contains implementations of the hyperboloid model and Orstein-Uhlenbeck process.

## Reproducing results in the main paper
- Figures in the main paper can be reproduced by running the Python scripts `make_figure_XX.py` (make sure to first change the working directory to be the semple folder).  
- Calculations reproducing results similar to Table 1 can be produced by running the script `energy_test.py` that is found in the "energy_test" branch of this repository (results are machine-dependent). Prior to launching the script (⚠️ you need an intel processor):
  - Go to branch "energy_test" of this repository and install pyjoules:
  ```commandline
  $pip install pyjoules
  ```
  - For each of bernoulli_glm.R, two_moons.R, hyperboloid.R, ornstein_uhlenbeck.R, set your working directory as the path of the project
  - Add the path of the GNU library packages at the end of energy_test.py
  - The results are saved in ./energy_results/
  
## Files explanation
model_name.R - runs SeMPLE with a specific model.

gaumixfit_sbibm.R - contains implementation of the SeMPLE algorithm in the form of the gaumixfit function.

compute_metrics.py - is used to compute performance metrics for a subfolder in the results directory, the metric results are saved in the same subfoler.

plot_results.py - contains functions to produce and save plots of results from results/ into figures/.

run_sbibm.py - is used to run the SNPE-C and SNL algorithms using SBIBM.

bic.R - is used to compute the BIC of the number of mixture components used in SeMPLE.
