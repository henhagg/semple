import sbibm
from sbibm.algorithms import snpe
from sbibm.algorithms import snle
import pandas as pd
import numpy as np
import os
import json

def save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation):
    settings_dict = {"num_simulations":num_simulations, "num_rounds":num_rounds, "num_samples":num_samples, "num_observation":num_observation}
    with open(f"{save_path}settings.json", "w") as outfile:
        json.dump(settings_dict, outfile)

def run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size):
    task = sbibm.get_task(task_name)
    if(algorithm=="snpe"):
        posterior_samples_list, _, _, elapsed_time = snpe(task=task, num_samples=num_samples, num_observation=num_observation, num_simulations=num_simulations, num_rounds=num_rounds, simulation_batch_size=simulation_batch_size)
    elif(algorithm=="snle"):
        posterior_samples_list, _, _, elapsed_time = snle(task=task, num_samples=num_samples, num_observation=num_observation, num_simulations=num_simulations, 
        num_rounds=num_rounds, simulation_batch_size=simulation_batch_size)
    else:
        raise Exception("Invalid algorithm")

    return posterior_samples_list, elapsed_time

def save_to_csv(posterior_samples_list, save_path):
    for idx, samples in enumerate(posterior_samples_list):
        sample_df = pd.DataFrame(samples.numpy())
        sample_df.to_csv(f"{save_path}post_sample_iter{idx+1}.csv", index=False, header=False)

def run_sbibm_obs(algorithm, task_name, observation_indices_list, num_simulations, num_rounds, num_samples, simulation_batch_size=1000, subfolder_save=""):
    for num_observation in observation_indices_list:
        print(num_observation)

        save_path = f"results/{task_name}/{algorithm}{subfolder_save}/obs{num_observation}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation)

        posterior_samples_list, elapsed_time = run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size)

        np.savetxt(f"{save_path}elapsed_time.csv", elapsed_time, delimiter =", ", fmt ='% s')
        
        save_to_csv(posterior_samples_list, save_path)
            
def run_sbibm_run(algorithm, task_name, run_indices_list, num_simulations, num_rounds, num_samples, simulation_batch_size=1000, subfolder_save="", num_observation=1):
    for run_index in run_indices_list:
        print(run_index)

        save_path = f"results/{task_name}/{algorithm}{subfolder_save}/run{run_index}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation)

        posterior_samples_list, elapsed_time = run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size)

        np.savetxt(f"{save_path}elapsed_time.csv", elapsed_time, delimiter =", ", fmt ='% s')
        
        save_to_csv(posterior_samples_list, save_path)

if __name__ == '__main__':
    print("running_sbibm")

    run_sbibm_obs(algorithm="snpe", task_name="mrna", observation_indices_list=range(1,5+1), num_simulations=30000, num_rounds=10, num_samples=10000, subfolder_save="/30k_10rounds")
    run_sbibm_obs(algorithm="snle", task_name="mrna", observation_indices_list=range(1,5+1), num_simulations=30000, num_rounds=10, num_samples=10000, subfolder_save="/30k_10rounds")

    # run_sbibm_obs(algorithm="snpe", task_name = "two_moons", observation_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k_10rounds")
    # run_sbibm_obs(algorithm="snle", task_name = "two_moons", observation_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k_10rounds")
    # run_sbibm_obs(algorithm="snpe", task_name = "two_moons", observation_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 4, num_samples = 10000, subfolder_save="/10k_4rounds")
    # run_sbibm_obs(algorithm="snle", task_name = "two_moons", observation_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 4, num_samples = 10000, subfolder_save="/10k_4rounds_resample")

    # run_sbibm_run(algorithm="snpe", task_name = "hyperboloid", run_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k_10rounds", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snle", task_name = "hyperboloid", run_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10rounds_resample", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snpe", task_name = "hyperboloid", run_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 4, num_samples = 10000, subfolder_save="/4rounds", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snle", task_name = "hyperboloid", run_indices_list=range(1,10+1), num_simulations = 10000, num_rounds = 4, num_samples = 10000, subfolder_save="/4rounds_resample", simulation_batch_size=1)

    # run_sbibm_obs(algorithm="snpe", task_name="bernoulli_glm", observation_indices_list=range(1,10+1), num_simulations=10000, num_rounds=2, num_samples=10000, subfolder_save="/5k")
    # run_sbibm_obs(algorithm="snle", task_name="bernoulli_glm", observation_indices_list=range(1,10+1), num_simulations=10000, num_rounds=2, num_samples=10000, subfolder_save="/5k")
    # run_sbibm_obs(algorithm="snpe", task_name="bernoulli_glm", observation_indices_list=range(1,10+1), num_simulations=10000, num_rounds=2, num_samples=10000, subfolder_save="/10rounds")
    # run_sbibm_obs(algorithm="snle", task_name="bernoulli_glm", observation_indices_list=range(1,10+1), num_simulations=10000, num_rounds=2, num_samples=10000, subfolder_save="/10rounds_resample")

    # run_sbibm_run(algorithm="snpe", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,10+1), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/10rounds", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snle", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,10+1), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/10rounds_resample", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snpe", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,10+1), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k_total", simulation_batch_size=1)
    # run_sbibm_run(algorithm="snle", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,10+1), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k_total", simulation_batch_size=1)