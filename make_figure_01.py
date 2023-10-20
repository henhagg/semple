import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sbibm
plt.style.use(Path.cwd() / Path('make_figure_stylesheet.txt'))

path_results = Path.cwd() / Path('results', 'two_moons')

n_rounds = 10
methods_list = ['semple/full_cov',
                f'snpe/10k_{n_rounds}rounds',
                f'snle/10k_{n_rounds}rounds',
                'reference']

posterior_samples = {}
for method in methods_list:
    posterior_samples[method] = {}
    for obs in [1, 8, 9]:
        if method != 'reference':
            filename = path_results / f'{method}/obs{obs}/post_sample_iter4.csv'
            posterior_samples[method][obs] = pd.read_csv(
                filename, header=None).values
        else:
            posterior_samples[method][obs] = sbibm.get_task(
                "two_moons").get_reference_posterior_samples(
                num_observation=obs).numpy()

methods_names = ['SeMPLE with 4 rounds',
                 'SNPE-C with 10 rounds',
                 'SNL 10 rounds',
                 'True posterior']
fig, ax = plt.subplots(
    figsize=(12.7, 9.6), nrows=3, ncols=4, sharex=True, sharey=True)
plt.subplots_adjust(
    wspace=0.15, hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.975)
for i, obs in enumerate([1, 8, 9]):
    for j, method in enumerate(methods_list):
        data = posterior_samples[method][obs]
        s = ax[i][j].hexbin(
            data[:,0],
            data[:,1],
            bins=25,
            extent=(-1, +1, -1, +1),
            cmap='viridis')
        ax[i][j].set_xlim(-1, +1)
        ax[i][j].set_ylim(-1, +1)
        ax[i][j].set_xticks([-1, -0.5, 0, +0.5, 1.0])
        ax[i][j].set_yticks([-1, -0.5, 0, +0.5, 1.0])
        if i == 0:
            ax[i][j].set_title(methods_names[j])

s.set_rasterized(True)
# plt.savefig('figure_01.pdf', format='pdf')
# plt.savefig('figure_01.png', format='png')
fig.show()