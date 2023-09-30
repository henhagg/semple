import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import json
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

def save_fig(save_path):
    plt.savefig(save_path)
#################################### METRICS ####################################################
def plot_metric_vs_sims(input_dir, metric_name, ylim=None):
    metric_df = pd.read_csv(f"{input_dir}/{metric_name}.csv")
    
    # sns.set_style("darkgrid")
    # sns.lineplot(data=metric_df, x="num_simulations", y=metric_name.upper(), style="algorithm", markers=True)

    plt.plot(metric_df["num_simulations"], metric_df[metric_name.upper()], marker="o")
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.show()

def plot_median_metric_vs_sims(input_dir, metric_name, num_obs_total, ylim=None):
    metric_df = pd.DataFrame()
    for num_observation in range(1,num_obs_total+1):
        metric_df_temp = pd.read_csv(f"{input_dir}/obs{num_observation}/{metric_name}.csv")
        metric_df = pd.concat([metric_df, metric_df_temp])

    sns.lineplot(data=metric_df, x="num_simulations", y=metric_name.upper(), err_style="bars",  estimator=lambda x:np.median(x), errorbar=("pi", 100), marker="o", markersize=7)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

def plot_multiple_algorithm_median_metric_vs_sims(input_dir_list, metric_name, obs_or_run, num_obs_or_run_total, show_fig=True, ylim=None, save_path=None):
    metric_df = pd.DataFrame()
    for input_dir in input_dir_list:
        for num_observation_or_run in range(1,num_obs_or_run_total+1):
            metric_df_temp = pd.read_csv(f"{input_dir}/{obs_or_run}{num_observation_or_run}/{metric_name}.csv")
            metric_df = pd.concat([metric_df, metric_df_temp])

    plt.figure(figsize=(8,6))
    sns.lineplot(data=metric_df, x="num_simulations", y=metric_name.upper(), err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100), style="algorithm", 
    hue="algorithm", markers=True, markersize=8, err_kws={"capsize":5})
    plt.xlabel("Number of simulations")

    if metric_name=="emdp2" or metric_name=="emdp1":
        plt.ylabel("Wasserstein distance")
    
    if ylim is not None:
        plt.ylim(ylim)
    
    if save_path is not None:
        save_fig(save_path)
    
    if show_fig:
        plt.show()

def plot_multiple_settings_median_metric_vs_sims(input_dir_list, legend_list, metric_name, obs_or_run, num_obs_or_run_total, show_fig=True, ylim=None, save_path=None):
    metric_df = pd.DataFrame()
    for idx, input_dir in enumerate(input_dir_list):
        for num_observation_or_run in range(1,num_obs_or_run_total+1):
            metric_df_temp = pd.read_csv(f"{input_dir}/{obs_or_run}{num_observation_or_run}/{metric_name}.csv")
            metric_df_temp["Settings"] = legend_list[idx]
            metric_df = pd.concat([metric_df, metric_df_temp])

    plt.figure(figsize=(8,6))
    sns.lineplot(data=metric_df, x="num_simulations", y=metric_name.upper(), err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100), style="Settings", 
    hue="Settings", markers=True, markersize=8, err_kws={"capsize":5})
    plt.xlabel("Number of simulations")

    if metric_name=="emdp2" or metric_name=="emdp1":
        plt.ylabel("Wasserstein distance")
    
    if ylim is not None:
        plt.ylim(ylim)
    
    if save_path is not None:
        save_fig(save_path)
    
    if show_fig:
        plt.show()

def plot_metric_vs_time(input_dir, metric_name, ylim=None):
    metric_df = pd.read_csv(f"{input_dir}/{metric_name}.csv")
    elapsed_time = pd.read_csv(f"{input_dir}/elapsed_time.csv", index_col=False, header=None)
 
    # plt.figure(figsize=(10,8))
    metric_df["elapsed_time"] = elapsed_time
    sns.lineplot(data=metric_df, x="elapsed_time", y=metric_name.upper(), marker="o")
    plt.ylabel(metric_name.upper(), fontsize=18)
    plt.xlabel("seconds", fontsize=18)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

def plot_multiple_algorithm_metric_vs_time(input_dir_list, metric_name, ylim=None):
    metric_df = pd.DataFrame()
    for input_dir in input_dir_list:
        metric_df_temp = pd.read_csv(f"{input_dir}/{metric_name}.csv")
        metric_df_temp["elapsed_time"] = pd.read_csv(f"{input_dir}/elapsed_time.csv", index_col=False, header=None)
        metric_df = pd.concat([metric_df, metric_df_temp])
    
    sns.lineplot(data=metric_df, x="elapsed_time", y=metric_name.upper(), marker="o", hue="algorithm")
    plt.ylabel(metric_name.upper(), fontsize=18)
    plt.xlabel("seconds", fontsize=18)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

def plot_aggregated_metric_vs_time(input_dir, obs_or_run, num_obs_or_run_total, metric_name, ylim=None):
    metric_df = pd.DataFrame()
    for num_observation_or_run in range(1,num_obs_or_run_total+1):
        metric_df_temp = pd.read_csv(f"{input_dir}/{obs_or_run}{num_observation_or_run}/{metric_name}.csv")
        metric_df_temp["elapsed_time"] = pd.read_csv(f"{input_dir}/{obs_or_run}{num_observation_or_run}/elapsed_time.csv", index_col=False, header=None)
        # metric_df = pd.concat([metric_df, metric_df_temp])
        sns.lineplot(data=metric_df_temp, x="elapsed_time", y=metric_name.upper(), marker="o")

    # sns.lineplot(data=metric_df, x="elapsed_time", y=metric_name.upper(), marker="o")
    plt.ylabel(metric_name.upper(), fontsize=18)
    plt.xlabel("seconds", fontsize=18)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
    
################################### POSTERIOR ####################################################
def plot_mc_semple(input_dir, param_index_list):
    with open(f"{input_dir}/settings.json", "r") as openfile:
        settings_dict = json.load(openfile)

    num_iters = settings_dict["num_iters"]

    posterior_samples_list = []
    for r in range(0,num_iters+1):
        posterior_samples_list.append(pd.read_csv(f"{input_dir}/post_sample_iter{r}.csv", index_col=False, header=None))

    for param_index in param_index_list:
        param_df = pd.DataFrame()
        for r, posterior_samples in enumerate(posterior_samples_list):
            mc_df = pd.DataFrame({"value": posterior_samples.iloc[:,param_index], "iteration": r, "param_index": param_index})
            param_df = pd.concat([param_df, mc_df])
        
        sns.lineplot(data=param_df, x=param_df.index, y="value", hue="iteration")
        plt.show()

def plot_pairs(file_path, save_path=None):
    posterior_samples = pd.read_csv(file_path, index_col=False, header=None)
    sns_plot = sns.pairplot(posterior_samples)
    plt.show()
    
    if save_path is not None:
        sns_plot.figure.savefig(f"{save_path}")

def plot_multiple_pairs(file_path_list, legend_list, parameter_name_list, fig_height=5, thin_interval=None, save_path=None, fig_show=True):
    df = pd.DataFrame()
    for idx, file_path in enumerate(file_path_list):
        if legend_list[idx] == "Reference":
            posterior_samples = pd.read_csv(file_path, index_col=False, header=None, skiprows=1) # Reference sample has header, the others don't
        else:
            posterior_samples = pd.read_csv(file_path, index_col=False, header=None)
        
        if thin_interval is not None:
            posterior_samples = posterior_samples.iloc[::thin_interval]
        
        posterior_samples.columns = parameter_name_list
            
        posterior_samples["Algorithm"] = legend_list[idx]
        df = pd.concat([df, posterior_samples], ignore_index=True)
    
    sns_plot = sns.pairplot(df, hue="Algorithm", height=fig_height, aspect= 1, plot_kws={"s": 5, "alpha":1}, diag_kws={"fill":False}, corner=False)
    # sns.kdeplot(df, x=0, hue="Algorithm")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        # sns_plot.figure.savefig(save_path)

    if fig_show:
        plt.show()

def plot_kde(file_path, parameter_name_list, save_path=None):
    posterior_samples = pd.read_csv(file_path, index_col=False, header=None)
    posterior_samples.columns = parameter_name_list
    sns.kdeplot(posterior_samples)
    plt.show()

def plot_multiple_algorithm_kde(file_path_list, legend_list, parameter_name_list, save_path=None, fig_show=True):
    df = pd.DataFrame()
    for idx, file_path in enumerate(file_path_list):
        if legend_list[idx] == "Reference":
            posterior_samples = pd.read_csv(file_path, index_col=False, header=None, skiprows=1) # Reference sample has header, the others don't
        else:
            posterior_samples = pd.read_csv(file_path, index_col=False, header=None)
        
        posterior_samples.columns = parameter_name_list
            
        posterior_samples["Algorithm"] = legend_list[idx]
        df = pd.concat([df, posterior_samples], ignore_index=True)
    
    fig, axes = plt.subplots(1, len(parameter_name_list), figsize=(15, 5), sharey=True, gridspec_kw={"wspace":0.1})
    plots = list()
    for idx, param in enumerate(parameter_name_list):
        sns_plot = sns.kdeplot(ax=axes[idx], data=df, x=param, hue="Algorithm", legend=True)
        plots.append(sns_plot)
    
    # handles, labels = axes[0].get_legend_handles_labels()
    # print(handles)
    # print(labels)
    # fig.legend(plots, parameter_name_list, loc='upper center')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        # sns_plot.figure.savefig(save_path)

    if fig_show:
        plt.show()

def plot_scatter(file_path_list, parameter_name_list, fig_size=(7,7), save_path=None, show_fig=True):
    for file_path in file_path_list:
        posterior_samples = pd.read_csv(file_path, index_col=False, header=None, skiprows=1) # skiprows if reference sample
        posterior_samples.columns = parameter_name_list
        # plt.scatter(posterior_samples[0], posterior_samples[1], s=0.5)

    plt.figure(figsize=fig_size)
    sns.scatterplot(data=posterior_samples, x=parameter_name_list[0], y=parameter_name_list[1], size=10, legend=False)
    if save_path is not None:
        save_fig(save_path)

    if show_fig:
        plt.show()

def using_mpl_scatter_density(fig, x, y, nrows, ncols, subfig_index, parameter_name_list, subfig_title=None, xlim=None, ylim=None, show_density_bar=True):
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    if subfig_title is not None:
        ax = fig.add_subplot(nrows, ncols, subfig_index , projection='scatter_density', title=subfig_title, xlabel=parameter_name_list[0], ylabel=parameter_name_list[1])
    else:
        ax = fig.add_subplot(nrows, ncols, subfig_index , projection='scatter_density', xlabel=parameter_name_list[0], ylabel=parameter_name_list[1])
    density = ax.scatter_density(x, y, cmap=white_viridis)
    
    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim)
    
    if subfig_index == ncols and show_density_bar:
        fig.colorbar(density, label='Number of points per pixel')

def plot_density_scatter(file_path, parameter_name_list, fig_size=None, save_path=None, show_fig=True, show_density_bar=True):
    posterior_samples = pd.read_csv(file_path, index_col=False).to_numpy()
    x = posterior_samples[:,0]
    y = posterior_samples[:,1]
    
    fig = plt.figure()
    if fig_size is not None:
        fig.set_size_inches(fig_size)
    using_mpl_scatter_density(fig, x, y, nrows=1, ncols=1, subfig_index=1, parameter_name_list=parameter_name_list, show_density_bar=show_density_bar)

    if save_path is not None:
        fig.savefig(save_path)

    if show_fig:
        plt.show()

def plot_multiple_density_scatter(file_path_list, subfig_titles, fig_size, xlim=None, ylim=None, save_path=None, show_fig=True):
    fig = plt.figure()
    fig.set_size_inches(fig_size)

    for idx, file_path in enumerate(file_path_list):
        posterior_samples = pd.read_csv(file_path, index_col=False).to_numpy()
        x = posterior_samples[:,0]
        y = posterior_samples[:,1]
        using_mpl_scatter_density(fig, x, y, nrows=1, ncols=len(file_path_list), subfig_index=idx+1, xlim=xlim, ylim=ylim, subfig_title=subfig_titles[idx])
    
    if show_fig:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path)

#################################### ACCEPTANCE RATE #############################################
def plot_semple_acceptance_rate(input_dir):    
    with open(f"{input_dir}/settings.json", "r") as openfile:
        settings_dict = json.load(openfile)

    num_iters = settings_dict["num_iters"]

    accrate = np.zeros(num_iters-1)
    for iter in range(2,num_iters+1):
        accrate[iter-2] = np.genfromtxt(f"{input_dir}/accrate_iter{iter}.csv", delimiter=',')

    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.plot(range(2,num_iters+1), accrate, marker='o')
    plt.ylim((0,1))
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Acceptance rate", fontsize=15)
    plt.show()

def plot_median_semple_acceptance_rate(input_dir, obs_or_run, num_obs_or_runs_total, save_path=None, show_fig=True):
    with open(f"{input_dir}/{obs_or_run}1/settings.json", "r") as openfile:
        settings_dict = json.load(openfile)
    num_iters = settings_dict["num_iters"]

    df = pd.DataFrame(columns=["iteration", "accrate"])
    for num_observation_or_run in range(1,num_obs_or_runs_total+1):
        for iter in range(2,num_iters+1):
            accrate = np.genfromtxt(f"{input_dir}/{obs_or_run}{num_observation_or_run}/accrate_iter{iter}.csv", delimiter=',').item()
            df.loc[len(df)] = {"iteration":iter, "accrate": accrate}
    
    plt.figure(figsize=(8,6))
    sns.lineplot(data=df, x="iteration", y="accrate", err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100), markers=True, markersize=8, err_kws={"capsize":5}, color="black")
    plt.ylim((0,1))
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Acceptance rate", fontsize=15)

    if save_path is not None:
        save_fig(save_path)
    
    if show_fig:
        plt.show()
    
def plot_multiple_median_semple_acceptance_rate(input_dir_list, legend_list, num_obs_total, save_path=None, show_fig=True):
    df = pd.DataFrame(columns=["iteration", "accrate", "Settings"])
    for idx, input_dir in enumerate(input_dir_list):
        with open(f"{input_dir}/obs1/settings.json", "r") as openfile:
            settings_dict = json.load(openfile)
        num_iters = settings_dict["num_iters"]
        
        for num_observation in range(1,num_obs_total+1):
            for iter in range(2,num_iters+1):
                accrate = np.genfromtxt(f"{input_dir}/obs{num_observation}/accrate_iter{iter}.csv", delimiter=',').item()
                df.loc[len(df)] = {"iteration":iter, "accrate": accrate, "Settings": legend_list[idx]}
    
    plt.figure(figsize=(8,6))
    sns.lineplot(data=df, x="iteration", y="accrate", err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100), style="Settings", hue="Settings", markers=True, markersize=8, err_kws={"capsize":5})
    plt.ylim((0,1))
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Acceptance rate", fontsize=15)

    if save_path is not None:
        save_fig(save_path)
    
    if show_fig:
        plt.show()

################################### TIME #########################################################
def plot_median_time_vs_iteration(input_dir, obs_or_run, num_obs_or_run_total):
    metric_df = pd.DataFrame()
    for num_obs_or_run in range(1,num_obs_or_run_total):
        # metric_df_temp = pd.read_csv(f"{input_dir}/{obs_or_run}{num_obs_or_run}/{metric_name}.csv")
        metric_df_temp = pd.DataFrame()
        metric_df_temp["elapsed_time"] = pd.read_csv(f"{input_dir}/{obs_or_run}{num_obs_or_run}/elapsed_time.csv", index_col=False, header=None)

        with open(f"{input_dir}/{obs_or_run}1/settings.json", "r") as openfile:
            settings_dict = json.load(openfile)
        if "num_iters" in settings_dict:
            metric_df_temp["iteration"] = range(1,settings_dict["num_iters"]+1)
        elif "num_rounds" in settings_dict:
            metric_df_temp["iteration"] = range(1,settings_dict["num_rounds"]+1)

        metric_df = pd.concat([metric_df, metric_df_temp])
    
    sns.lineplot(data=metric_df, x="iteration", y="elapsed_time", marker="o", err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100))
    plt.ylabel("Elapsed time")
    plt.xlabel("Iteration")
    plt.show()

def plot_multiple_algorithm_median_time_vs_iteration(input_dir_list, legend_list, obs_or_run, num_obs_or_run_total, save_path=None, show_fig=True):
    metric_df = pd.DataFrame()
    for idx, input_dir in enumerate(input_dir_list):
        for num_obs_or_run in range(1,num_obs_or_run_total):
            # metric_df_temp = pd.read_csv(f"{input_dir}/{obs_or_run}{num_obs_or_run}/{metric_name}.csv")
            metric_df_temp = pd.DataFrame()
            metric_df_temp["elapsed_time"] = pd.read_csv(f"{input_dir}/{obs_or_run}{num_obs_or_run}/elapsed_time.csv", index_col=False, header=None)
            metric_df_temp["elapsed_time"] = metric_df_temp["elapsed_time"]/60

            metric_df_temp["Algorithm"] = legend_list[idx]

            with open(f"{input_dir}/{obs_or_run}1/settings.json", "r") as openfile:
                settings_dict = json.load(openfile)
            if "num_iters" in settings_dict:
                metric_df_temp["iteration"] = range(1,settings_dict["num_iters"]+1)
            elif "num_rounds" in settings_dict:
                metric_df_temp["iteration"] = range(1,settings_dict["num_rounds"]+1)

            metric_df = pd.concat([metric_df, metric_df_temp])
    
    sns.lineplot(data=metric_df, x="iteration", y="elapsed_time", marker="o", hue="Algorithm", style="Algorithm", err_style="bars", estimator=lambda x:np.median(x), errorbar=("pi", 100))
    plt.ylabel("Elapsed time [minutes]")
    plt.xlabel("Iteration")

    if save_path is not None:
        plt.savefig(save_path)
    
    if show_fig:
        plt.show()


if __name__ == '__main__':
    print("plot_results")

    #################################### TWO MOONS ##########################################################
    # plot_pairs(file_path="results/two_moons/semple/K80/obs1/post_sample_iter4.csv")
    # plot_kde(file_path="results/two_moons/semple/final/obs1/post_sample_iter4.csv", parameter_name_list=[r"$\theta_1$", r"$\theta_2$"])
    # plot_scatter(file_path_list=[f"../sbibm/sbibm/tasks/two_moons/files/num_observation_1/reference_posterior_samples.csv.bz2"], fig_size=(6,6), parameter_name_list=[r"$\theta_1$", r"$\theta_2$"],
    #              save_path="figures/two_moons/posterior_example.pdf")
    # plot_density_scatter(file_path=f"../sbibm/sbibm/tasks/two_moons/files/num_observation_1/reference_posterior_samples.csv.bz2", parameter_name_list=[r"$\theta_1$", r"$\theta_2$"],
    #                         save_path="figures/two_moons/posterior_example_density.pdf", fig_size=(5,5), show_density_bar=False)

    # plot_multiple_algorithm_kde(file_path_list=[f"results/two_moons/snle/obs1/post_sample_iter4.csv", f"results/two_moons/snpe/obs1/post_sample_iter4.csv", 
    #     f"results/two_moons/semple/final/obs1/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/two_moons/files/num_observation_1/reference_posterior_samples.csv.bz2"],
    #     legend_list=["SNL", "SNPE-C", "semple", "Reference"], parameter_name_list=[r"$\theta_1$", r"$\theta_2$"])

    # for num_observation in range(1,11):
    #     plot_multiple_pairs(file_path_list=[f"results/two_moons/snle/obs{num_observation}/post_sample_iter4.csv", f"results/two_moons/snpe/obs{num_observation}/post_sample_iter4.csv", 
    #         f"results/two_moons/semple/final/obs{num_observation}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/two_moons/files/num_observation_{num_observation}/reference_posterior_samples.csv.bz2"],
    #         legend_list=["SNL", "SNPE-C", "semple", "Reference"], parameter_name_list=[r"$\theta_1$", r"$\theta_2$"], fig_height=3, fig_show=False, thin_interval=1,
    #         save_path=f"figures/two_moons/pairplot/observation{num_observation}.png")

    # for num_observation in range(1,11):
    #     plot_multiple_density_scatter(file_path_list=[f"results/two_moons/semple/final/obs{num_observation}/post_sample_iter4.csv", f"results/two_moons/snpe/obs{num_observation}/post_sample_iter4.csv",
    #         f"results/two_moons/snle/obs{num_observation}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/two_moons/files/num_observation_{num_observation}/reference_posterior_samples.csv.bz2"],
    #         subfig_titles=["semple", "SNPE-C", "SNL", "Reference"], fig_size=(20,5), xlim=(-1.1,1.1), ylim=(-1.1,1.1), save_path=f"figures/two_moons/density_scatter/observation{num_observation}.pdf", show_fig=False)

    # plot_metric_vs_time(input_dir="results/two_moons/semple/K80/obs1", metric_name="c2st")
    # plot_metric_vs_sims(input_dir="results/two_moons/semple/obs2", metric_name="c2st")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=["results/two_moons/semple/final", "results/two_moons/snpe", "results/two_moons/snle"], 
    #         metric_name=metric_name, num_obs_total=10, save_path=f"figures/two_moons/algorithm_{metric_name}_vs_sims.pdf")

    for metric_name in ["c2st", "emdp2", "mmd"]:
        plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=["results/two_moons/semple/full_cov", "results/two_moons/snpe/", "results/two_moons/snle"], 
            metric_name=metric_name, num_obs_total=10, save_path=f"figures/two_moons/algorithm_{metric_name}_vs_sims.pdf")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/two_moons/semple/final", "results/two_moons/semple/MHpost", "results/two_moons/semple/full_cov"],
    #         legend_list=["Surrogate likelihood", "Surrogate posterior", "Full covariance matrix"], metric_name=metric_name, num_obs_total=10, save_path=f"figures/two_moons/settings_{metric_name}_vs_sims.pdf")
    
    # plot_semple_acceptance_rate(input_dir="results/two_moons/semple/final/obs1")
    # plot_median_semple_acceptance_rate(input_dir="results/two_moons/semple/final", num_obs_total=10, save_path="figures/two_moons/semple_median_accrate.pdf")
    # plot_multiple_median_semple_acceptance_rate(input_dir_list=["results/two_moons/semple/final/", "results/two_moons/semple/MHpost"], 
    #     legend_list=["Surrogate likelihood", "Surrogate posterior"], num_obs_total=10, save_path="figures/two_moons/semple_median_accrate_lik_post.pdf")

    # plot_multiple_algorithm_median_time_vs_iteration(input_dir_list=["results/two_moons/semple/final", "results/two_moons/snpe", "results/two_moons/snle"],
    #                                                     legend_list=["semple", "SNPE", "SNLE"], obs_or_run="obs", num_obs_or_run_total=10, save_path="figures/two_moons/time_vs_iter.pdf")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/two_moons/semple/full_cov", "results/two_moons/semple/full_cov_keepD0"],
    #         legend_list=["Discard prior predictive D0", "Keep prior predictive D0"], metric_name=metric_name, obs_or_run="obs", num_obs_or_run_total=2, save_path=f"figures/two_moons/keepD0_{metric_name}_vs_sims.png")

    ##################################################### HYPERBOLOID ############################################
    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=["results/hyperboloid/semple/10k", "results/hyperboloid/snpe/4rounds", "results/hyperboloid/snle/4rounds"], 
    #         metric_name=metric_name, obs_or_run="run", num_obs_or_run_total=5, save_path=f"figures/hyperboloid/algorithm_{metric_name}_vs_sims.pdf")

    # plot_median_semple_acceptance_rate(input_dir="results/hyperboloid/semple/final", obs_or_run="run", num_obs_or_runs_total=5, show_fig=True, save_path="figures/hyperboloid/semple_accrate.pdf")

    # for num_run in range(1,6):
    #     plot_multiple_density_scatter(file_path_list=[f"results/hyperboloid/semple/10k/run{num_run}/post_sample_iter4.csv", f"results/hyperboloid/snpe/4rounds/run{num_run}/post_sample_iter4.csv",
    #         f"results/hyperboloid/snle/4rounds/run{num_run}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/hyperboloid/files/num_observation_1/reference_posterior_samples.csv.bz2"],
    #         subfig_titles=["semple", "SNPE-C", "SNL", "Reference"], fig_size=(20,5), xlim=(-2,2), ylim=(-2,2), save_path=f"figures/hyperboloid/density_scatter/run{num_run}.pdf", show_fig=True)

    # for num_run in range(1,6):
    #     plot_multiple_pairs(file_path_list=[f"results/hyperboloid/snle/4rounds/run{num_run}/post_sample_iter4.csv", f"results/hyperboloid/snpe/4rounds/run{num_run}/post_sample_iter4.csv", 
    #         f"results/hyperboloid/semple/10k/run{num_run}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/hyperboloid/files/num_observation_1/reference_posterior_samples.csv.bz2"],
    #         legend_list=["SNL", "SNPE-C", "semple", "Reference"], parameter_name_list=[r"$\theta_1$", r"$\theta_2$"], fig_height=3, thin_interval=10, fig_show=True,
    #         save_path=f"figures/hyperboloid/pairplot/run{num_run}_thin10.png")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/hyperboloid/semple/final", "results/hyperboloid/semple/10k"],
    #         legend_list=["20k prior", "10k prior"], metric_name=metric_name, obs_or_run="run", num_obs_or_run_total=10)#, save_path=f"figures/two_moons/settings_{metric_name}_vs_sims.pdf")

    # plot_median_time_vs_iteration(input_dir="results/hyperboloid/semple/10k", obs_or_run="run", num_obs_or_run_total=5)
    # plot_multiple_algorithm_median_time_vs_iteration(input_dir_list=["results/hyperboloid/semple/10k", "results/hyperboloid/snpe/4rounds", "results/hyperboloid/snle/4rounds"],
    #                                                     legend_list=["semple", "SNPE", "SNLE"], obs_or_run="run", num_obs_or_run_total=5, save_path="figures/hyperboloid/time_vs_iter.pdf")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/hyperboloid/semple/10k", "results/hyperboloid/semple/10k_keepD0"],
    #         legend_list=["Discard prior predictive D0", "Keep prior predictive D0"], metric_name=metric_name, obs_or_run="run", num_obs_or_run_total=2, save_path=f"figures/hyperboloid/keepD0_{metric_name}_vs_sims.png")

    ################################################### ORNSTEIN-UHLENBECK ########################################
    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=["results/ornstein_uhlenbeck/semple/10k_prior", "results/ornstein_uhlenbeck/snpe/40k_total", "results/ornstein_uhlenbeck/snle/40k_total"], 
    #         metric_name=metric_name, obs_or_run="run", num_obs_or_run_total=10, save_path=f"figures/ornstein_uhlenbeck/algorithm_{metric_name}_vs_sims.pdf")
        
    # plot_median_semple_acceptance_rate(input_dir="results/ornstein_uhlenbeck/semple/10k_prior", obs_or_run="run", num_obs_or_runs_total=5, show_fig=True, save_path="figures/ornstein_uhlenbeck/semple_median_accrate.pdf")

    # for num_run in range(1,6):
    #     plot_multiple_pairs(file_path_list=[f"results/ornstein_uhlenbeck/snle/40k_total/run{num_run}/post_sample_iter4.csv", f"results/ornstein_uhlenbeck/snpe/40k_total/run{num_run}/post_sample_iter4.csv", 
    #         f"results/ornstein_uhlenbeck/semple/10k_prior/run{num_run}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/ornstein_uhlenbeck/files/num_observation_1/reference_posterior_samples.csv.bz2"],
    #         legend_list=["SNL", "SNPE-C", "semple", "Reference"], parameter_name_list=[r"$\alpha$", r"$\beta$", r"$\sigma$"], fig_height=3, fig_show=True, thin_interval=10,
    #         save_path=f"figures/ornstein_uhlenbeck/pairplot/run{num_run}_thin10.png")

    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/ornstein_uhlenbeck/semple/10k_prior", "results/ornstein_uhlenbeck/semple/final", "results/ornstein_uhlenbeck/semple/20k_prior_K20"],
    #         legend_list=["10k prior", "final", "20k prior"], metric_name=metric_name, obs_or_run="run", num_obs_or_run_total=3)#, save_path=f"figures/two_moons/settings_{metric_name}_vs_sims.pdf")

    # plot_multiple_algorithm_median_time_vs_iteration(input_dir_list=["results/ornstein_uhlenbeck/semple/10k_prior", "results/ornstein_uhlenbeck/snpe/40k_total", "results/ornstein_uhlenbeck/snle/40k_total"],
    #                                                  legend_list=["semple", "SNPE-C", "SNL"], obs_or_run="run", num_obs_or_run_total=5, save_path="figures/ornstein_uhlenbeck/time_vs_iter.pdf")

    # TEST
    # plot_metric_vs_time(input_dir="results/ornstein_uhlenbeck/semple/final/run1", metric_name="c2st")
    # plot_aggregated_metric_vs_time(input_dir="results/ornstein_uhlenbeck/semple/final", obs_or_run="run", num_obs_or_run_total=10, metric_name="c2st")
    # plot_multiple_algorithm_metric_vs_time(input_dir_list=["results/ornstein_uhlenbeck/semple/10k_prior/run2", "results/ornstein_uhlenbeck/snpe/40k_total/run2", "results/ornstein_uhlenbeck/snle/40k_total/run2"], metric_name="c2st")
    # plot_median_time_vs_iteration(input_dir="results/ornstein_uhlenbeck/semple/10k_prior", metric_name="c2st", obs_or_run="run", num_obs_or_run_total=3)

    ################################################## SLCP #######################################################
    # for metric_name in ["c2st", "emdp2", "mmd"]:
    #     plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=["results/slcp/semple/80k", "results/slcp/snpe/4rounds", "results/slcp/snle/4rounds"], 
    #         metric_name=metric_name, obs_or_run="obs", num_obs_or_run_total=3, save_path=f"figures/slcp/algorithm_{metric_name}_vs_sims.pdf")
        
    # plot_median_semple_acceptance_rate(input_dir="results/slcp/semple/80k", obs_or_run="obs", num_obs_or_runs_total=3, show_fig=True, save_path="figures/slcp/semple_median_accrate.pdf")

    # for num_observation in range(1,2):
    #     plot_multiple_pairs(file_path_list=[f"results/slcp/semple/80k/obs{num_observation}/post_sample_iter4.csv", f"results/slcp/snpe/4rounds/obs{num_observation}/post_sample_iter4.csv",
    #         f"results/slcp/snle/4rounds/obs{num_observation}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/slcp/files/num_observation_{num_observation}/reference_posterior_samples.csv.bz2"],
    #         legend_list=["semple", "SNPE-C", "SNL", "Reference"], parameter_name_list=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"], fig_height=2,
    #         fig_show=True, thin_interval=10, save_path=f"figures/slcp/pairplot/obs{num_observation}_thin10.png")

    # plot_multiple_algorithm_median_time_vs_iteration(input_dir_list=["results/slcp/semple/80k", "results/slcp/snpe/4rounds", "results/slcp/snle/4rounds"],
    #                                                     legend_list=["semple", "SNPE-C", "SNL"], obs_or_run="obs", num_obs_or_run_total=3, save_path="figures/slcp/time_vs_iter.pdf")

    # for num_observation in range(1,2):
    #     plot_multiple_algorithm_kde(file_path_list=[f"results/slcp/semple/80k/obs{num_observation}/post_sample_iter4.csv", f"results/slcp/snpe/4rounds/obs{num_observation}/post_sample_iter4.csv",
    #         f"results/slcp/snle/4rounds/obs{num_observation}/post_sample_iter4.csv", f"../sbibm/sbibm/tasks/slcp/files/num_observation_{num_observation}/reference_posterior_samples.csv.bz2"],
    #         legend_list=["semple", "SNPE-C", "SNL", "Reference"], parameter_name_list=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"], save_path="figures/slcp/multi_kde.pdf"

    ############################################# TWISTED ########################################################
    # plot_pairs(file_path=f"results/twisted/semple/run1/post_sample_iter1.csv")

    ############################################# BERNOULLI GLM ##################################################
    # plot_metric_vs_sims(input_dir="results/bernoulli_glm/semple/umberto/obs1", metric_name="c2st", ylim=[0,1])
    # plot_metric_vs_sims(input_dir="results/bernoulli_glm/semple/umberto/obs2", metric_name="c2st", ylim=[0,1])
    # plot_median_metric_vs_sims(input_dir="results/bernoulli_glm/semple/2k_2iter/", metric_name="c2st", num_obs_total=3, ylim=[0,1])
    # plot_median_metric_vs_sims(input_dir="results/bernoulli_glm/semple/5k_2iter/", metric_name="c2st", num_obs_total=3, ylim=[0,1])
    # plot_median_metric_vs_sims(input_dir="results/bernoulli_glm/semple/10k_2iter/", metric_name="c2st", num_obs_total=3, ylim=[0,1])
    
    # plot_mc_semple(input_dir="results/bernoulli_glm/semple/10k_2iter/obs3", param_index_list=[0])

    # plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/bernoulli_glm/semple/2k_2iter/", "results/bernoulli_glm/semple/5k_2iter/", "results/bernoulli_glm/semple/10k_2iter/"], 
    #                                              legend_list=["2.5k", "5k", "10k"], metric_name="c2st", obs_or_run="obs", num_obs_or_run_total=3, ylim=[0,1])

    # plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/bernoulli_glm/semple/2k_2iter/", "results/bernoulli_glm/semple/5k_2iter/", "results/bernoulli_glm/semple/10k_2iter/"], 
                                                #  legend_list=["2.5k", "5k", "10k"], metric_name="emdp2", obs_or_run="obs", num_obs_or_run_total=3)
    
    # plot_multiple_settings_median_metric_vs_sims(input_dir_list=["results/bernoulli_glm/semple/2k_2iter/", "results/bernoulli_glm/semple/5k_2iter/", "results/bernoulli_glm/semple/10k_2iter/"], 
                                                #  legend_list=["2.5k", "5k", "10k"], metric_name="mmd", obs_or_run="obs", num_obs_or_run_total=3)

    # for num_observation in range(1,2):
    #     plot_multiple_pairs(file_path_list=[f"results/bernoulli_glm/semple/2k_2iter/obs{num_observation}/post_sample_iter1.csv", f"results/bernoulli_glm/semple/2k_2iter/obs{num_observation}/post_sample_iter2.csv", 
    #                                         f"../sbibm/sbibm/tasks/bernoulli_glm/files/num_observation_{num_observation}/reference_posterior_samples.csv.bz2"],
    #         legend_list=["SeMPLE iter1 (2500 simulations)", "SeMPLE iter2 (5000 simulations)", "Reference"], parameter_name_list=[f"parameter_{i}" for i in range(1,11)], fig_height=1.2, thin_interval=10,
    #         fig_show=True)

    # for settings in ["2k", "5k", "10k"]:
    #     for metric_name in ["c2st", "emdp2", "mmd"]:
    #         plot_multiple_algorithm_median_metric_vs_sims(input_dir_list=[f"results/bernoulli_glm/semple/{settings}_2iter", f"results/bernoulli_glm/snpe/{settings}", f"results/bernoulli_glm/snle/{settings}"], 
    #             metric_name=metric_name, obs_or_run="obs", num_obs_or_run_total=3, save_path=f"figures/bernoulli_glm/algorithm_{metric_name}_vs_sims_{settings}.pdf")
    
    # for settings in ["2k", "5k", "10k"]:
    #     plot_multiple_algorithm_median_time_vs_iteration(input_dir_list=[f"results/bernoulli_glm/semple/{settings}_2iter", f"results/bernoulli_glm/snpe/{settings}", f"results/bernoulli_glm/snle/{settings}"],
    #                                                     legend_list=["SeMPLE", "SNPE-C", "SNL"], obs_or_run="obs", num_obs_or_run_total=3, save_path=f"figures/bernoulli_glm/time_vs_iter_{settings}.pdf")