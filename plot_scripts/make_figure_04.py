import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import pandas as pd
import sbibm
from scipy.ndimage import gaussian_filter1d
plt.style.use(Path.cwd() / Path('plot_scripts/make_figure_stylesheet.txt'))

path_results = Path.cwd() / Path('results', 'hyperboloid')
methods_list = ['semple/10k', 'snpe/40k_10rounds', 'snle/10rounds_resample']

posterior_samples = {}
for method in methods_list:
    if method == 'semple/10k':  # plot SeMPLE results from round 4
        filename = path_results / f'{method}/run{5}/post_sample_iter4.csv'
    else:
        filename = path_results / f'{method}/run{5}/post_sample_iter10.csv'
    posterior_samples[method] = pd.read_csv(
        filename, header=None).values

posterior_samples['reference'] = sbibm.get_task(
                "hyperboloid").get_reference_posterior_samples(
                num_observation=1).numpy()

theta_1_ref = pd.read_csv(
    path_results / f'reference/hyperboloid-theta1-400.csv', header=None).values[:,0]
marg_1_ref = pd.read_csv(
    path_results / f'reference/hyperboloid-true-marg1.csv', header=None).values[:,0]
joint_ref = pd.read_csv(
    path_results / f'reference/hyperboloid-joint-400x400.csv', header=None).values

# figsize=(9.6, 9.6)
fig = plt.figure(figsize=(6.95, 6.95))
gs = GridSpec(
    nrows=4,
    ncols=4,
    left=0.10,
    right=0.95,
    top=0.95,
    bottom=0.10,
    hspace=0.25,
    wspace=0.25)
ax_marg1 = fig.add_subplot(gs[0:2,0:2])
ax_marg2 = fig.add_subplot(gs[2:4,2:4])

ax_pair = {}
ax_pair['reference'] = fig.add_subplot(gs[0,2])
ax_pair['semple/10k'] = fig.add_subplot(gs[0,3])
ax_pair['snpe/40k_10rounds'] = fig.add_subplot(gs[1,2])
ax_pair['snle/10rounds_resample'] = fig.add_subplot(gs[1,3])

colors = ['black', 'C0', 'C1', 'C2']
methods_names = [
    'True posterior', 'SeMPLE with 4 rounds', 'SNPE-C with 10 rounds', 'SNL with 10 rounds']

for m, method in enumerate(['reference'] + methods_list):

    if method in ['reference']:
        ax_marg1.plot(
            theta_1_ref, marg_1_ref, color=colors[m], label=methods_names[m])
        ax_marg2.plot( # symmetry
            theta_1_ref, marg_1_ref, color=colors[m], label=methods_names[m])
        
    elif method in ['semple/10k', 'snpe/40k_10rounds']:
        
        hist, edges_1 = np.histogram(posterior_samples[method][:,0], bins=50)
        edges_1 = (edges_1[1:] + edges_1[:-1])/2
        delta = edges_1[1] - edges_1[0]
        pdf_1 = hist / (np.sum(hist) * delta)
        pdf_1 = gaussian_filter1d(pdf_1, sigma=2) # smoothing the histogram

        hist, edges_2 = np.histogram(posterior_samples[method][:,1], bins=50)
        edges_2 = (edges_2[1:] + edges_2[:-1])/2
        delta = edges_2[1] - edges_2[0]
        pdf_2 = hist / (np.sum(hist) * delta)
        pdf_2 = gaussian_filter1d(pdf_2, sigma=2) # smoothing the histogram

        ax_marg1.plot(edges_1, pdf_1, color=colors[m], label=methods_names[m])
        ax_marg2.plot(edges_2, pdf_2, color=colors[m], label=methods_names[m])

    elif method in ['snle/10rounds_resample']:

        ax_marg1.plot([], [], color=colors[m], label=methods_names[m])

    ax_marg1.set_xlim(-2.2, +2.2)
    ax_marg1.set_ylim(0, 0.55)
    ax_marg1.set_xlabel(r'$\theta_1$', fontsize=14)
    ax_marg1.text(
        x=-2.1,
        y=0.5,
        s=r'$p(\theta_1 | \mathbf{y}_0)$',
        fontsize=14,
        ha='left')

    ax_marg2.set_xlim(-2.2, +2.2)
    ax_marg2.set_ylim(0, 0.55)
    ax_marg2.set_xlabel(r'$\theta_2$', fontsize=14)
    ax_marg2.text(
        x=-2.1,
        y=0.5,
        s=r'$p(\theta_2 | \mathbf{y}_0)$',
        fontsize=14,
        ha='left')

    ax_marg2.text(
        x=0,
        y=1.185,
        s=r'$p(\theta_1, \theta_2 | \mathbf{y}_0)$',
        fontsize=14,
        ha='center')

    if method in methods_list:

        ax_pair[method].scatter(
            posterior_samples[method][::5,0],
            posterior_samples[method][::5,1],
            s=5,
            alpha=0.50,
            color=colors[m])     
        
    elif method in ['reference']:
        T1 = np.reshape(joint_ref[:,0], (400, 400))
        T2 = np.reshape(joint_ref[:,1], (400, 400))
        Z = np.reshape(joint_ref[:,2], (400, 400))
        ax_pair[method].contour(T1, T2, Z, levels=10, colors='black')

    ax_pair[method].set_xticks([])
    ax_pair[method].set_yticks([])

ax_pair['snle/10rounds_resample'].set_xlabel(r'$\theta_1$')
ax_pair['snle/10rounds_resample'].yaxis.set_label_position("right")
ax_pair['snle/10rounds_resample'].set_ylabel(r'$\theta_2$')
leg = ax_marg1.legend(loc=(0.12, -0.75))

# plt.savefig('figure_04_v1.pdf', format='pdf')
fig.show()