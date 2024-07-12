# Sequential Mixture Posterior and Likelihood Estimation (SeMPLE)
SeMPLE is a framework for simulation-based inference that learns surrogate models for the likelihood function and the posterior distribution of model parameters, and outputs draws from the posterior. It provides a faster and lightweight solution compared to samplers using neural networks for density estimation, while retaining accuracy.

## Setup
SeMPLE is written in R. To run SeMPLE you need to install the xLLiM R package that gives access to the GLLiM function. Follow the instructions below, do NOT use the version provided on CRAN.

Run the following command in R to get the correct version of xLLiM (requires the R package "devtools" to be installed): 
```commandline
# install.packages("devtools")
devtools::install_github("epertham/xLLiM", ref = "master")
```

Set the working directory to be the repository semple folder.

## Running SeMPLE
To run SeMPLE with with an implemented model/task run the corresponding `model_name.R` script in the `run_scripts` folder, e.g. `run_scripts/two_moons.R`. The algorithm outputs are saved into the results directory.

## Add new model/task
To implement a new model to run SeMPLE with create a new model file according to `models/model_name/model_name_model.R`. The model file need to define four functions, the `model` function to simulate data, the `prior_pdf` function that defines the probability density function of the prior distribution, the `sample_prior` function that returns a sample from the prior distribution and the `jacobian` function that defines the optional transformation jacobian. Additionally, at least one observed data set needs to be added to `models/model_name/num_observation_X`.

## SBIBM
To run SNPE-C or SNL to produce results to be compared to SeMPLE results, install the SBIBM fork https://github.com/henhagg/sbibm. It contains modifications to output posterior samples from each SNPE-C and SNL algorithm iteration instead of only from the last algorithm iteration. Additionally, it is modified to output the runtime of SNPE-C and SNL. Furthermore, the fork contains implementations of the hyperboloid model, the Orstein-Uhlenbeck process and the mRNA model.

## Reproducing results in the main paper
- The algorithm results can be reproduced by using the scripts in the `run_scripts` folder. The results presented in the paper are available in the `results` folder.
- Performance metric computations can be reproduced by running the `src/compute_metrics.py` script.
- Figures in the main paper can be reproduced by running the scripts `plot_scripts` folder (make sure to first change the working directory to be the semple folder).
- Calculations reproducing results similar to Table 1 can be produced by running the script `energy_test.py` that is found in the "energy_test" branch of this repository (results are machine-dependent). Prior to launching the script (⚠️ you need an intel processor):
  - Go to branch "energy_test" of this repository and install pyjoules:
  ```commandline
  $pip install pyjoules
  ```
  - For each of bernoulli_glm.R, two_moons.R, hyperboloid.R, ornstein_uhlenbeck.R, set your working directory as the path of the project
  - Add the path of the GNU library packages at the end of energy_test.py
  - The results are saved in ./energy_results/
  
## Repository structure
- The `run_scripts` folder contains `model_name.R` scripts to run SeMPLE with the implemented models. The `run_sbibm.py` script can be used to run SNPE-C and SNL.

- The `src` folder contains the `semple.R` script with the SeMPLE algorithm impelementation. It also contains the `bic.R` script that computes the Bayesian Information Criterion (BIC) to help determine the number of mixture components K to initialize SeMPLE with. The `compute_metrics.py` script can be used to compute performance metrics of algorithm posterior samples relative to a reference posterior.

- The `plot_scripts` folder contains scripts to plot the figures presented in the paper.

- The `results` folder contains the results presented in the paper. New results from running SeMPLE are saved in this folder.