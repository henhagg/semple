import sys
sys.path.append('./sbibm')

import sbibm
from sbibm.algorithms import snpe
from sbibm.algorithms import snle
from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.csv_handler import CSVHandler

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import tracemalloc
import pandas as pd
import numpy as np
import os
import time
import json

def save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation):
    settings_dict = {"num_simulations":num_simulations, "num_rounds":num_rounds, "num_samples":num_samples, "num_observation":num_observation}
    with open(f"{save_path}settings.json", "w") as outfile:
        json.dump(settings_dict, outfile)

def run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size, ctx):
    task = sbibm.get_task(task_name)
    if(algorithm=="snpe"):
        posterior_samples_list, _, _, elapsed_time = snpe(task=task, num_samples=num_samples, num_observation=num_observation, num_simulations=num_simulations, num_rounds=num_rounds, simulation_batch_size=simulation_batch_size, ctx=ctx)
    elif(algorithm=="snle"):
        posterior_samples_list, _, _, elapsed_time = snle(task=task, num_samples=num_samples, num_observation=num_observation, num_simulations=num_simulations, num_rounds=num_rounds, simulation_batch_size=simulation_batch_size, ctx=ctx)
    else:
        raise Exception("Invalid algorithm")

    return posterior_samples_list, elapsed_time

def save_to_csv(posterior_samples_list, save_path):
    for idx, samples in enumerate(posterior_samples_list):
        sample_df = pd.DataFrame(samples.numpy())
        sample_df.to_csv(f"{save_path}post_sample_iter{idx+1}.csv", index=False, header=False)

def run_sbibm_obs(algorithm, task_name, observation_indices_list, num_simulations, num_rounds, num_samples, simulation_batch_size=1000, subfolder_save="", ctx=None, num_observation=None):
    peaks = []
    for num_observation in observation_indices_list:
        print(num_observation)

        save_path = f"results/{task_name}/{algorithm}{subfolder_save}/obs{num_observation}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation)

        tracemalloc.start()

        posterior_samples_list, elapsed_time = run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size, ctx)
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

        np.savetxt(f"{save_path}elapsed_time.csv", elapsed_time, delimiter =", ", fmt ='% s')
        
        save_to_csv(posterior_samples_list, save_path)
    return np.median(peaks)
            
def run_sbibm_run(algorithm, task_name, run_indices_list, num_simulations, num_rounds, num_samples, simulation_batch_size=1000, subfolder_save="", num_observation=1, ctx=None):
    peaks = []
    for run_index in run_indices_list:
        print(run_index)

        save_path = f"results/{task_name}/{algorithm}{subfolder_save}/run{run_index}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_settings(save_path, num_simulations, num_rounds, num_samples, num_observation)

        tracemalloc.start()
        posterior_samples_list, elapsed_time = run_algorithm(algorithm, task_name, num_samples, num_observation, num_simulations, num_rounds, simulation_batch_size, ctx)
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

        np.savetxt(f"{save_path}elapsed_time.csv", elapsed_time, delimiter =", ", fmt ='% s')
        
        save_to_csv(posterior_samples_list, save_path)
    return np.median(peaks)

if __name__ == '__main__':
    ########### SNLE et SNPE methods ###########
    
    peaks = []
    for k in range(9):
        tracemalloc.start()
        csv_handler = CSVHandler(f'./energy_results/result_{k}.csv')
        with EnergyContext(handler=csv_handler, start_tag=f'exp_{k}') as ctx:
        
            if k == 0:
                time.sleep(2)

            if k == 1:             
                peak = run_sbibm_obs(algorithm="snle", task_name = "two_moons", observation_indices_list=range(1,11), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k", ctx = ctx)

            if k == 2:
                peak = run_sbibm_obs(algorithm="snpe", task_name = "two_moons", observation_indices_list=range(1,11), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k", ctx = ctx)

            if k == 3:
                peak = run_sbibm_run(algorithm="snle", task_name = "hyperboloid", run_indices_list=range(1,11), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k", simulation_batch_size=1, ctx = ctx)

            if k == 4:
                peak = run_sbibm_run(algorithm="snpe", task_name = "hyperboloid", run_indices_list=range(1,11), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k", simulation_batch_size=1, ctx = ctx)

            if k == 5:
                peak = run_sbibm_obs(algorithm="snle", task_name = "bernoulli_glm", observation_indices_list=range(1,11), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k", ctx = ctx)

            if k == 6:
                peak = run_sbibm_obs(algorithm="snpe", task_name = "bernoulli_glm", observation_indices_list=range(1,11), num_simulations = 10000, num_rounds = 10, num_samples = 10000, subfolder_save="/10k", ctx = ctx)

            if k == 7:
                peak = run_sbibm_run(algorithm="snpe", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,11), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k_total", simulation_batch_size=1, ctx = ctx)

            if k == 8:
                peak = run_sbibm_run(algorithm="snle", task_name = "ornstein_uhlenbeck", run_indices_list=range(1,11), num_simulations = 40000, num_rounds = 10, num_samples = 10000, subfolder_save="/40k_total", simulation_batch_size=1, ctx = ctx)
        
        csv_handler.save_data() 
        if k != 0: peaks.append(peak)

    pd.DataFrame({'exp': [k for k in range(1, 9)], 'peaks': peaks}).to_csv('./energy_results/peak_mem.csv')

    ########### SEMPLE ###########

    r = robjects.r
    path_gnu = "your path to gnu library"
    importr('xLLiM', lib_loc=path_gnu)
    importr('SimDesign', lib_loc=path_gnu)
    importr('withr', lib_loc=path_gnu)
    importr('ggplot2', lib_loc=path_gnu)
    importr('mixtools', lib_loc=path_gnu)
    importr('tictoc', lib_loc=path_gnu)
    importr('rjson', lib_loc=path_gnu)
    importr('mvtnorm', lib_loc=path_gnu)
    importr('mcmc', lib_loc=path_gnu)
    importr('SFSI', lib_loc=path_gnu)
    
    task_names = ['bernoulli_glm', 'hyperboloid', 'ornstein_uhlenbeck', 'two_moons']
    peaks_semple = []
    for task_name in task_names:
        csv_handler = CSVHandler(f"./energy_results/{task_name}.csv")
        with EnergyContext(handler=csv_handler, start_tag=f'{task_name}_1') as ctx:
                peaks = []
                for k in range(1, 11):
                    r['source'](task_name + '.R')
                    tracemalloc.start()
                    robjects.r.run(k)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peaks.append(peak)
                    ctx.record(tag=f"{task_name}_{k+1}")
                peaks_semple.append(np.median(peaks))
            csv_handler.save_data()
        pd.DataFrame({'exp': task_names, 'peaks': peaks_semple}).to_csv('./energy_results/peak_mem_semple.csv')
