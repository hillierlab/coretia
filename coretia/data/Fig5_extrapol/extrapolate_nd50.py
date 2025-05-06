''' Demonstrates precision of extrapolation of Hill fitting to non-measured dilution range.
Interestingly, weighting to the difference across responses does not modify the fit.
If MCMC sigma is using weights (other than technical repeat variability), the uncertainty of
ND50 estimation gets substantially larger - which likely is a reflection of a more realistic
estimate, i.e. ~0 samples limit uncertainty but actually do not yield information to guide the fit.

As the trueND50 gets further away from sampled dilutions, the estimated nd50 gets more and more biased
to the right: this is due to the top=1 constraint.
'''

import numpy as np
import fire
import matplotlib.pyplot as plt
# Update default rcParams to hide top and right spines
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
from coretia.bootstrap import annotate_nd50_curve
from coretia.data import add_noise
from coretia.bayesian import hill_curve
from coretia.plot import create_custom_legend, plot_dpi
from pathlib import Path

ms = 4 # marker size for scatter
plot_height = 6


def plot_initial_curve(ax, dilutions_broad, dilutions_6, true_curve, noisy_curve, n_repeats, cv, true_ND50, alpha=1, s=10):
    """Plot the initial Hill curve and sampled points with noise."""
    ax.plot(dilutions_broad, true_curve, label=f"Mean of Simulated\nFull Data (ND50={1 / true_ND50})", color="black", alpha=alpha, linestyle=(0, (1, 10)))  # loosely dotted
    ax.scatter(np.repeat(dilutions_6, n_repeats), noisy_curve.flatten(), color='black', label=f"Points used, CV={cv}", alpha=alpha, s=s)
    ax.set_xscale('log', base=2)
    ax.set_xticks(dilutions_broad)
    ax.set_xticklabels([f"1/{int(1/d)}" for d in dilutions_broad])
    ax.set_xlabel('Dilution')
    ax.set_ylabel('Response')
    ax.legend(frameon=False)
    #ax.set_title('Initial Hill Curve Fit and Estimated Dilution Points')


def init_data(nd50_titer = 2**6, hill_slope=1, cv=0.1, n_repeats=3):
    # Dilution series (log2 scale)
    dilutions_broad = np.array([2 ** -i for i in range(2, 9)])  # 1/4, 1/8 ...

    # Narrower dilution range, sampling only part of the true Hill curve
    dilutions_6 = np.array([2 ** -i for i in range(2, 5)])  # 1/4, 1/8 ... 1/128

    # True parameters for sample 1 (ND50 and Hill slope)
    true_ND50 = 1 / nd50_titer
    top, bottom = 1, 0

    # Simulate % neutralization data for sample 1
    true_curve = hill_curve(dilutions_broad, hill_slope, true_ND50, top, bottom)

    noisy_broad_curve = np.column_stack([add_noise(true_curve, cv) for _ in range(n_repeats)])

    return dilutions_broad, dilutions_6, true_curve, noisy_broad_curve, nd50_titer, cv


def nd50_mean_CI_converge(out_path=None, true_ND50=2**6, n_comparisons = 20, seed=np.random.seed(41), dpi=plot_dpi):
    from scipy.stats import ttest_rel
    # Run the MCMC model
    from coretia.nl_model import hill_fit_cached as hill_fit_mcmc
    from coretia.bayesian import extrapolate_mcmc
    alpha=0.3

    nd50_mlh_all = {}
    max_technical_repeats = 3
    tested_repeats = range(max_technical_repeats,0,-1)
    n_repeats_large = n_comparisons * max_technical_repeats
    dilutions_broad, dilutions_narrow, true_curve, noisy_broad_curve, div1, cv =  \
        init_data(nd50_titer=true_ND50, hill_slope=1, cv=0.1, n_repeats=n_repeats_large)

    fig1, ax = plt.subplots(figsize=(6, plot_height))
    matching_indices = np.isin(dilutions_broad, dilutions_narrow)
    noisy_curve = noisy_broad_curve[matching_indices]
    plot_initial_curve(ax, dilutions_broad, dilutions_narrow, true_curve, noisy_curve, n_repeats_large, cv, true_ND50, alpha=alpha)
    ax.plot(dilutions_narrow, noisy_curve.mean(axis=1), 'b', alpha=alpha)

    for n_repeats in tested_repeats:
        nd50_mlh_all[str(n_repeats)] = []
        # Sample n_repeats from the high-repeat base matrix
        for nr in range(n_comparisons):
            cols = np.random.choice(n_repeats_large, n_repeats, replace=False)
            noisy_curve = noisy_broad_curve[matching_indices][:, cols]
            trace = hill_fit_mcmc(dilutions_narrow, noisy_curve, use_weights=False)
            mlh = extrapolate_mcmc(trace, 0.5)[:3]
            nd50_mlh_all[str(n_repeats)].append(mlh)


    nd50_means_all = {k1: [v2[0] for v2 in v1] for k1, v1 in nd50_mlh_all.items()}
    nd50_v_ci_widths = {k1: [np.diff(v2[1:])[0] for v2 in v1] for k1, v1 in nd50_mlh_all.items()}

    # Compute pairwise p-values using a paired t-test
    import itertools
    for desc, data in zip(['Mean ND50', 'CI'],[nd50_means_all, nd50_v_ci_widths]):
        print(f'{desc}:')
        for key1, key2 in itertools.combinations(data.keys(), 2):
            stat, p_value = ttest_rel(data[key1], data[key2])
            print(f"Paired t-test between {key1}-reps and {key2}-reps: t={stat:.4f}, p={p_value:.4f}, diff={np.mean(data[key1])-np.mean(data[key2])}")

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=""))
    ax.set_ylabel('Transduction Efficiency (%)')
    # set_xscale adds xticks with log2 steps, set xticks explicitly to only include actual dilution points
    ax.set_ylim(0, 1)

    colors_nd50 = {'1': 'tab:cyan', '2': 'tab:brown', '3': 'tab:pink'}
    f_kw = {'nd50_y_position_on_plot': 0.05,
            'hill_error_style': [''],  #
            'mcmc_show_nd50': 0,
            'ylabel': 'Transduction Efficiency (%)'
            }
    for i1, (k1, v1) in enumerate(nd50_mlh_all.items()):
        v2 = np.array(v1)[:,0]  # only keep ND50 estimates, not CI for each estimate
        mlh = [v2.mean()]
        mlh.extend(np.percentile(v2, [2.5, 97.5]))  # CI_low, and CI_high of ND50 estimates
        annotate_nd50_curve(mlh[0], mlh[1], mlh[2], ax, 1, colors_nd50[k1], f_kw)
        f_kw['nd50_y_position_on_plot'] += i1*0.03


    legend0 = zip([f'Synthetic ground truth curve ND50=1/{true_ND50}',
                   'Noisy samples (technical replicates)',
                   f'Mean of all ({n_repeats_large})\ntechnical replicates'],
                  ['black', 'black', 'blue'], [':','dot','-'])
    bbox0 = (0.725, 0.86)
    legend1 = zip(colors_nd50.keys(), colors_nd50.values(), ['--'] * 3)
    bbox1 = (0.685, 0.70)
    create_custom_legend([legend0, legend1], titles=['', 'Random replicates used\nfor one ND50 estimate:'], bbox=[bbox0, bbox1], alpha=alpha)
    fig1.tight_layout()
    if out_path is not None:
        fig1.savefig(out_path / Path(__file__).name.replace('.py',f"_nd50{true_ND50}_extrapol{n_comparisons}.png"), dpi=dpi)


if __name__ == '__main__':
    fire.Fire()