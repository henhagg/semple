import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sbibm
from sbibm.metrics import c2st
import torch
from tqdm import tqdm
plt.style.use(Path.cwd() / Path('make_figure_stylesheet.txt'))

path_results = Path.cwd() / Path('results', 'two_moons')

methods_list = ['semple/full_cov', 'snpe/10k_10rounds', 'snle/10k_10rounds']
observation_list = range(1, 10+1)

mtr = []
eti = []
for method in methods_list:
    for obs in observation_list:
        filename = path_results / f'{method}/obs{obs}/c2st.csv'
        mtr.append(pd.read_csv(filename))
        filename = path_results / f'{method}/obs{obs}/elapsed_time.csv'
        eti_mo = pd.read_csv(filename, header=None, names=['time'])
        eti_mo['num_observation'] = obs
        eti_mo['algorithm'] = method
        if method == 'semple/full_cov':
            eti_mo['num_iter'] = list(range(1, 4+1))
        else:
            eti_mo['num_iter'] = list(range(1, 10+1))
        eti.append(eti_mo)        
mtr_df = pd.concat(mtr)
eti_df = pd.concat(eti)

mtr_df.loc[mtr_df['algorithm'] == 'JASS', 'algorithm'] = 'semple/full_cov'
mtr_df.loc[mtr_df['algorithm'] == 'SNPE', 'algorithm'] = 'snpe/10k_10rounds'
mtr_df.loc[mtr_df['algorithm'] == 'SNLE', 'algorithm'] = 'snle/10k_10rounds'

c2st = {}
for method in methods_list:
    c2st[method] = {}
    c2st[method]['median'] = mtr_df.loc[mtr_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_simulations').median()['C2ST'].values
    c2st[method]['lower'] = mtr_df.loc[mtr_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_simulations').min()['C2ST'].values
    c2st[method]['upper'] = mtr_df.loc[mtr_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_simulations').max()['C2ST'].values

elapsed_time = {}

for method in methods_list:
    elapsed_time[method] = {}
    elapsed_time[method]['median'] = eti_df.loc[eti_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_iter').median()['time'].values/60
    elapsed_time[method]['lower'] = eti_df.loc[mtr_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_iter').min()['time'].values/60
    elapsed_time[method]['upper'] = eti_df.loc[mtr_df['algorithm'] == method].drop(
        'algorithm', axis=1).groupby(
        'num_iter').max()['time'].values/60

methods_names = ['SeMPLE with 4 rounds', 'SNPE-C with 10 rounds', 'SNL with 10 rounds', 'True posterior']

fig, ax = plt.subplots(figsize=(12.7, 5.5), ncols=2)
plt.subplots_adjust(top=0.95, left=0.10,  right=0.90)
colors = ['C0', 'C1', 'C2']

for m, method in enumerate(methods_list):
    if method == 'semple/full_cov':
        x = 2500 * np.arange(1, 4+1)
    else:
        x = 1000 * np.arange(1, 10+1)
    y = c2st[method]['median']
    ax[0].plot(x, y, label=methods_names[m], c=colors[m])
    ax[0].scatter(x, y, s=20, c=colors[m])
    yinf = c2st[method]['lower']
    ysup = c2st[method]['upper']
    ax[0].fill_between(x, yinf, ysup, alpha=0.20)
ax[0].set_ylim(0.50, 0.85)
ax[0].set_xticks([2500, 5000, 7500, 10_000])
ax[0].legend(loc='upper right')
ax[0].set_xlabel('Number of simulations')
ax[0].set_ylabel('C2ST')

for m, method in enumerate(methods_list):
    if method == 'semple/full_cov':
        x = 2500 * np.arange(1, 4+1)
    else:
        x = 1000 * np.arange(1, 10+1)
    y = elapsed_time[method]['median']
    ax[1].plot(x, y, label=methods_names[m], c=colors[m])
    ax[1].scatter(x, y, s=20, c=colors[m])
    yinf = elapsed_time[method]['lower']
    ysup = elapsed_time[method]['upper']
    ax[1].fill_between(x, yinf, ysup, alpha=0.20)
ax[1].set_xticks([2500, 5000, 7500, 10_000])
ax[1].set_xlabel('Number of simulations')
ax[1].set_ylabel('Cumulative elapsed time [minutes]')

# plt.savefig('figure_02.pdf', format='pdf')
fig.show()