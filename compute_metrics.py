import pandas as pd
import sbibm
from sbibm.metrics import c2st
from sbibm.metrics import mmd
import torch
import ot
import numpy as np
import json

def calc_emd(ref_data_set, data_set, p=2, numItermax=100_000):
    n = ref_data_set.shape[0]
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    if p == 2:
        M = ot.dist(ref_data_set, data_set)
        return np.sqrt(ot.emd2(a, b, M, numItermax=numItermax))
    elif p == 1:
        M = ot.dist(ref_data_set, data_set, metric="euclidean")
        return ot.emd2(a, b, M, numItermax=numItermax)

def compute_sbibm_metric(metric_name, algorithm, task_name, input_dir, num_observation):
    task = sbibm.get_task(task_name)
    
    with open(f"{input_dir}/settings.json", "r") as openfile:
        settings_dict = json.load(openfile)

    if(algorithm == "jass"):
        num_simulation_per_iteration = settings_dict["num_simulation_per_iteration"]
        num_priorpred_samples = settings_dict["num_priorpred_samples"]
        num_iters = settings_dict["num_iters"]
        num_simulations_list = [num_priorpred_samples + (num_simulation_per_iteration * i) for i in range(num_iters)]
    else:
        num_simulations_total = settings_dict["num_simulations"]
        num_rounds = settings_dict["num_rounds"]
        num_iters = num_rounds
        num_observation = settings_dict["num_observation"]
        num_simulations_list = [round(num_simulations_total/num_rounds * i) for i in range(1,num_rounds+1)]

    metric_df = pd.DataFrame(columns=['num_simulations', 'num_observation', 'algorithm', metric_name.upper()])

    for iter in range(1,num_iters+1):
        print(iter)
        posterior_samples = pd.read_csv(f"{input_dir}/post_sample_iter{iter}.csv", index_col=False, header=None)
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)

        if(metric_name == "c2st"):
            posterior_samples_tensor = torch.tensor(posterior_samples.values)
            metric_score = c2st(reference_samples, posterior_samples_tensor).item()
        elif(metric_name == "emdp1"):
            metric_score = calc_emd(reference_samples.numpy(), posterior_samples.to_numpy(), p=1, numItermax=1_000_000)
        elif(metric_name == "emdp2"):
            metric_score = calc_emd(reference_samples.numpy(), posterior_samples.to_numpy(), p=2, numItermax=1_000_000)
        elif(metric_name == "mmd"):
            posterior_samples_tensor = torch.tensor(posterior_samples.values)
            metric_score = mmd(reference_samples, posterior_samples_tensor).item()
        else:
            raise Exception("Invalid metric")

        metric_df.loc[len(metric_df)] = {"num_simulations":num_simulations_list[iter-1], "num_observation":num_observation, "algorithm":algorithm.upper(), metric_name.upper():metric_score}

    metric_df.to_csv(f"{input_dir}/{metric_name}.csv", index=False, header=True)

def compute_nrmse(algorithm, input_dir, task_name, dim_param, num_observation, num_runs):
    metric_name = "nrmse"

    true_parameters = pd.read_csv(f"../sbibm/sbibm/tasks/{task_name}/files/num_observation_{num_observation}/true_parameters.csv", index_col=False, header=True).values

    with open(f"{input_dir}/run1/settings.json", "r") as openfile:
        settings_dict = json.load(openfile)

    if(algorithm == "jass"):
        num_simulation_per_iteration = settings_dict["num_simulation_per_iteration"]
        num_priorpred_samples = settings_dict["num_priorpred_samples"]
        num_iters = settings_dict["num_iters"]
        num_simulations_list = [num_priorpred_samples + (num_simulation_per_iteration * i) for i in range(num_iters)]
    else:
        num_simulations_total = settings_dict["num_simulations"]
        num_rounds = settings_dict["num_rounds"]
        num_iters = num_rounds
        num_observation = settings_dict["num_observation"]
        num_simulations_list = [round(num_simulations_total/num_rounds * i) for i in range(1,num_rounds+1)]

    metric_df = pd.DataFrame(columns=['num_simulations', 'num_observation', 'algorithm', metric_name.upper()])

    for iter_index, num_simulations in num_simulations_list.items():
        print(iter_index, num_simulations)
        rmse_sum = np.zeros(dim_param)
        for run_index in range(1,num_runs+1):
            posterior_samples = pd.read_csv(f"{input_dir}/run{run_index}/post_sample_iter{iter}.csv", index_col=False, header=None)
            posterior_samples = posterior_samples.to_numpy()
            posterior_means = np.mean(posterior_samples, axis=0)
            rmse_sum = rmse_sum + (posterior_means-true_parameters)**2
        rmse = np.sqrt(rmse_sum/num_runs)
        nrmse = np.mean(rmse/true_parameters)
        
        metric_df.loc[len(metric_df)] = {"num_simulations":num_simulations, "num_observation":num_observation, "algorithm":algorithm.upper(), metric_name.upper():nrmse}

    metric_df.to_csv(f"{input_dir}/{metric_name}.csv", index=False, header=True)

def compute_sbibm_metric_multiple_obs(metric_name, task_name, algorithm, num_obs_range, subfolder=""):
    for num_observation in num_obs_range:
        print(num_observation)
        compute_sbibm_metric(metric_name=metric_name, task_name=task_name, algorithm=algorithm, input_dir=f"results/{task_name}/{algorithm}{subfolder}/obs{num_observation}", num_observation=num_observation)

def compute_sbibm_metric_multiple_runs(metric_name, task_name, algorithm, run_index_range, num_observation, subfolder=""):
    for run_index in run_index_range:
        print(run_index)
        compute_sbibm_metric(metric_name=metric_name, task_name=task_name, algorithm=algorithm, input_dir=f"results/{task_name}/{algorithm}{subfolder}/run{run_index}", num_observation=num_observation)

def compute_multiple_sbibm_metrics_multiple_obs(metric_name_list, task_name, algorithm, num_obs_range, subfolder=""):
    for metric_name in metric_name_list:
        print(metric_name)
        compute_sbibm_metric_multiple_obs(metric_name=metric_name, task_name=task_name, algorithm=algorithm, num_obs_range=num_obs_range, subfolder=subfolder)

def compute_multiple_sbibm_metrics_multiple_runs(metric_name_list, task_name, algorithm, run_index_range, subfolder=""):
    for metric_name in metric_name_list:
        print(metric_name)
        compute_sbibm_metric_multiple_runs(metric_name=metric_name, task_name=task_name, algorithm=algorithm, run_index_range=run_index_range, subfolder=subfolder, num_observation=1)

if __name__ == '__main__':
    print("computing metrics")
    compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st"], task_name = "bernoulli_glm", algorithm = "jass", num_obs_range=range(1,2), subfolder = "/5k")

    # compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "two_moons", algorithm = "jass", num_obs_range=range(1,11), subfolder = "/K80")
    # compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "two_moons", algorithm = "jass", num_obs_range=range(1,11), subfolder = "/full_cov")
    
    # compute_multiple_sbibm_metrics_multiple_runs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "hyperboloid", algorithm = "jass", run_index_range=range(1,11), subfolder = "/10k")
    # compute_multiple_sbibm_metrics_multiple_runs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "hyperboloid", algorithm = "snpe", run_index_range=range(6,11), subfolder = "/4rounds")
    
    # compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "slcp", algorithm = "jass", num_obs_range=range(1,4), subfolder = "/80k")
    # compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "slcp", algorithm = "snpe", num_obs_range=range(5,11), subfolder = "/4rounds")
    # compute_multiple_sbibm_metrics_multiple_obs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "slcp", algorithm = "snle", num_obs_range=range(5,11), subfolder = "/4rounds")

    # compute_multiple_sbibm_metrics_multiple_runs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "ornstein_uhlenbeck", algorithm = "jass", run_index_range=range(4,11), subfolder = "/10k_prior")
    # compute_multiple_sbibm_metrics_multiple_runs(metric_name_list = ["c2st", "emdp2", "mmd"], task_name = "ornstein_uhlenbeck", algorithm = "snpe", run_index_range=range(6,11), subfolder = "/40k_total")

