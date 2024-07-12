import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sbibm

plt.style.use(Path.cwd() / Path('plot_scripts/make_figure_stylesheet.txt'))

fig, ax = plt.subplots(
    figsize=(8.8, 6.7), nrows=3, ncols=4, sharex=True, sharey=True)
plt.subplots_adjust(
    wspace=0.15, hspace=0.15, left=0.10, right=0.95, top=0.90, bottom=0.10)

mincnt = 1

for i, obs in enumerate([1, 2, 3]):

    column_01 = f'./results/two_moons/semple/full_cov/obs{obs}/post_sample_iter4.csv'
    df_01 = pd.read_csv(column_01, names=['theta_0', 'theta_1'])
    column_02 = f'./results/two_moons/snpe/10k_10rounds/obs{obs}/post_sample_iter10.csv'
    df_02 = pd.read_csv(column_02, names=['theta_0', 'theta_1'])
    column_03 = f'./results/two_moons/snle/10k_10rounds/obs{obs}/post_sample_iter10.csv'
    df_03 = pd.read_csv(column_03, names=['theta_0', 'theta_1'])
    # column_04 = f'./twomoons_semple/obs{obs}/reference_posterior_samples.csv'
    # df_04 = pd.read_csv(column_04, header=0, names=['theta_0', 'theta_1'])
    samples = sbibm.get_task("two_moons").get_reference_posterior_samples(num_observation=obs).numpy()
    df_04 = pd.DataFrame(data=samples, columns=['theta_0', 'theta_1'])

    ax[i][0].hexbin(
        df_01['theta_0'], df_01['theta_1'], gridsize=(100, 100), bins=100,
        extent=(-1, 1, -1, 1), mincnt=mincnt)
    ax[i][1].hexbin(
        df_02['theta_0'], df_02['theta_1'], gridsize=(100, 100), bins=100,
        extent=(-1, 1, -1, 1), mincnt=mincnt)
    ax[i][2].hexbin(
        df_03['theta_0'], df_03['theta_1'], gridsize=(100, 100), bins=100,
        extent=(-1, 1, -1, 1), mincnt=mincnt)
    ax[i][3].hexbin(
        df_04['theta_0'], df_04['theta_1'], gridsize=(100, 100), bins=100,
        extent=(-1, 1, -1, 1), mincnt=mincnt)

    for j in range(3):
        ax[i][j].set_xlim(-1, 1)
        ax[i][j].set_ylim(-1, 1)
        ax[i][j].set_xticks([-1, 0, +1])
        ax[i][j].set_xticklabels([-1, 0, +1])
        ax[i][j].set_yticks([-1, 0, +1])
        ax[i][j].set_yticklabels([-1, 0, +1])

ax[0][0].set_title('SeMPLE with 4 rounds', fontsize=10)
ax[0][1].set_title('SNPE-C with 10 rounds', fontsize=10)
ax[0][2].set_title('SNL with 10 rounds', fontsize=10)
ax[0][3].set_title('True posterior', fontsize=10)

ax[0][0].text(x=-1.55, y=0, s='observation 1', rotation=90, ha='center', va='center')
ax[1][0].text(x=-1.55, y=0, s='observation 2', rotation=90, ha='center', va='center')
ax[2][0].text(x=-1.55, y=0, s='observation 3', rotation=90, ha='center', va='center')

fig.show()
# plt.savefig('figure_01.pdf', format='pdf')
# plt.savefig('figure_01.png', format='png')
