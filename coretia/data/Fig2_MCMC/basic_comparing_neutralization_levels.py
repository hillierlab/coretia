"""
Demonstrates fitting Hill curve to non-neutralizing, mildly and neutralizing samples.
Determines which sample is non-neutralizing.
"""
import numpy as np
import fire
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.ERROR)
from coretia.tutorials import generate_synthetic_data
from coretia.bootstrap import (visualize_mcmc_fit, nd50_barplot_mcmc, bootstrap_nd50_lin_2_dilutions,
                                      annotate_nd50_curve, add_fraction_ticks)
from coretia.bayesian import extrapolate_mcmc, linear_nd50_slope_in_log2_space_estimates, \
    hill_curve
from coretia.nl_model import hill_fit_cached as hill_fit_mcmc
from coretia.plot import create_custom_legend, linear_bootstrap_zoomed, title_within_axes, plot_dpi
from pathlib import Path
import multiprocessing
import platform
cpu_count = multiprocessing.cpu_count()
n_cores = 1 if platform.system() == 'Windows' else cpu_count-1
import coretia.data
data_dir = Path(coretia.data.__file__).parent

colors_nd50_methods = {'Non-statistical': 'tab:cyan', 'Linear-bootstrap': 'tab:brown', 'Hill-MCMC': 'tab:pink'}

plot_height = 6
line_width = 2
offset = (line_width / plot_dpi) * 5
alpha = 0.4

nd50_stats_dict = {'plot_kw':{}}

# Function to fit and compare models, with an additional check on Hill fit
def fit_model(x_data, y_curve):
    nd50, slope = linear_nd50_slope_in_log2_space_estimates(x_data, np.median(y_curve, axis=1))

    # Fit Hill curve model
    hill_trace = hill_fit_mcmc(x_data, y_curve)

    nd50_samples = hill_trace.posterior['nd50'].values.flatten()

    # Compute the 95% credible interval for the difference
    lower_bound, upper_bound = np.percentile(nd50_samples, [2.5, 97.5])

    # Check if the credible interval for nd50 suggests a flat or neutralizing curve
    if lower_bound < 0.5:
        return hill_trace, True
    else:
        return hill_trace, False


def compare_same_nd50_different_cvs(out_path=None, cv=[0.1, 0.2], dilution_step = -1, seed=42, nd50_thr_log = 0, barplot_yaxis=1):
    seed = np.random.seed(seed)
    nd50_stats_dict = {'plot_kw': {
        'title': '',
        'xlabel': 'Serum Dilution',
        'ylabel': 'Transduction Efficiency (%)'
    }}
    x_data = 2. ** np.arange(-2, -9, dilution_step)  # 7 dilutions
    curves_cv_pair = [generate_synthetic_data(x_data, 1, 3, neutralization_type='strong', cv=cv1)[0] for cv1 in cv]

    # Perform fitting and evaluation on each curve
    cv_colors = ['green', 'olive']
    fig1, ax = plt.subplots(figsize=(5, plot_height))
    f_kw = {'nd50_y_position_on_plot': 0.05,
            'hill_error_style': ['hill_CI95_bars'],
            'mcmc_show_nd50': 0,}
    cvs = [str(cv1) for cv1 in cv]
    nd50 = {c2:nd50_3_methods(ax, color, f_kw,x_data, y_curve, nd50_stats_dict=nd50_stats_dict) for i, (y_curve, c2, color) in enumerate(zip(curves_cv_pair, cvs, cv_colors))}

    legends = []
    titles = []
    if len(cv) > 1:
        legends.append(zip([f'CV = {c1:1.1f}' for c1 in cv], cv_colors, ['-'] * len(colors)))
        bbox = [(0.715, 0.79),(0.75, 0.62), (0.725, 0.55)]
        titles.append('Simulated\nSerum Data')
        ccolor = ['gray', 'gray']
    else:
        bbox = [(0.715, 0.87),(0.733, 0.76)]
        ccolor = [cv_colors[0]] *2
    legends.append(zip(["Raw Samples", "Hill-fit"],
                      ccolor,
                      [':', '-']
                      ))
    legends.append(zip(colors_nd50_methods.keys(), colors_nd50_methods.values(), ['--'] * 3))
    titles.extend(['','ND50 estimation:'])
    create_custom_legend(legends, titles=titles, bbox=bbox, alpha=alpha)
    plt.tight_layout()
    plt.gca().set_ylim(0, 1.19)
    if out_path is not None:
        plt.gca().figure.savefig(out_path / (Path(__file__).name[:-3] + f"_hill_fit_{cv}_{dilution_step}_{seed}.png"), dpi=plot_dpi)
    plt.close()

    #####################
    # Barplot
    plot_kw = {'figures': {'mcmc_nd50_pair_with_next': 1,
                           'mcmc_barplot_significance': 'star',
                           'label_rotation': 0, 'label_ha': 'center',
                           'barplotsize': (2, plot_height), 'mcmc_barplot_significance_positions': 'max',
                           'barplot_text_gap': 0.11,
                           'barplot_toffset': 0.022,
                           'barplotylim': 32,
                           'barplot_yaxis':barplot_yaxis,
                           }
               }
    plot_kw['xlabel'] = 'Serum Dilution'
    # nd50_thr_log=0 is the most stringent check for statistical similarity
    fix = nd50_barplot_mcmc(['Hill-MCMC']*len(cv), {c1:nd50[c1]['stats']['Hill-MCMC'] for c1 in cvs}, {c1:nd50[c1]['samples']['Hill-MCMC'] for c1 in cvs},
                            'dilution', colors=cv_colors, plot_kw=plot_kw, nd50_thr_log=nd50_thr_log)
    plt.tight_layout()
    if out_path is not None:
        fix.savefig(out_path / \
                Path(__file__).name.replace('.py', f"_barplot_{cv}_{dilution_step}_{seed}.png"), dpi=plot_dpi)

    #### Barplot with Hill-MCMC and linear bootstrap 
    if len(cv)>1:
        plot_kw['figures']['barplot_category_minor_xticks'] = 1
    labels = ['Non-statistical','Linear-boostrap','Hill-MCMC']
    plot_kw['figures']['nd50_labels'] = labels
    plot_kw['figures']['label_rotation'] = 40
    plot_kw['figures']['label_ha'] = 'right'
    plot_kw['figures']['barplot_text_gap'] = 0.17
    nd50_merge = {f'Non-statistical {k1}': nd50[k1]['stats']['Non-statistical'] for k1 in cvs}
    nd50_merge.update({f'Linear {k1}': nd50[k1]['stats']['Linear-bootstrap'] for k1 in cvs})
    nd50_merge.update({f'Hill-MCMC {k1}': nd50[k1]['stats']['Hill-MCMC'] for k1 in cvs})

    nd50samples_merge = {f'Non-statistical {k1}': None for k1 in cvs}
    nd50samples_merge.update({f'Linear {k1}': nd50[k1]['samples']['Linear-bootstrap'] for k1 in cvs})
    nd50samples_merge.update({f'Hill-MCMC {k1}': nd50[k1]['samples']['Hill-MCMC'] for k1 in cvs})
    fix = nd50_barplot_mcmc(['Non-statistical','Linear-boostrap','Hill-MCMC'], nd50_merge, nd50samples_merge, 'dilution',
                            colors=list(colors_nd50_methods.values())*3,
                            plot_kw=plot_kw, nd50_thr_log=0, categories=cv)
    title_within_axes(f"θ = {nd50_thr_log}\nlog2 units", fontsize=8)
    plt.tight_layout()
    if out_path is not None:
        fix.savefig(out_path / \
                    Path(__file__).name.replace('.py', f"_barplot_bootstrap_{cv}_{dilution_step}_{seed}_thr{nd50_thr_log}.png"), dpi=plot_dpi)


def nd50_3_methods(ax, color, f_kw, x_data, y_curve, d_c='dilution', nd50_stats_dict = None):
    if ax is None:
        ax = plt.gca()
    nd50 = {'stats': {'Hill-MCMC': {}, 'Linear-bootstrap': {}, 'Non-statistical': {}},
            'samples': {'Hill-MCMC': {}, 'Linear-bootstrap': {}}
            }

    method = 'Hill-MCMC'
    nd50['stats'][method], nd50['samples'][method] = mcmc_fit_plot(
        alpha, ax, color, offset, x_data, y_curve, d_c=d_c, f_kw=f_kw, nd50_stats_dict=nd50_stats_dict
    )
    m, low, high = nd50['stats'][method]
    f_kw['nd50_y_position_on_plot'] = 0.05
    annotate_nd50_curve(m, low, high, ax, 1, colors_nd50_methods[method], f_kw, nd50_limit=1/4 if d_c=='dilution' else 3.13)

    method = 'Linear-bootstrap'
    nd50['samples'][method] = 2 ** bootstrap_nd50_lin_2_dilutions(
        np.log2(x_data), np.array(y_curve), avg_f=np.mean
    )
    x5 = nd50['samples'][method].mean()
    low_high = np.percentile(nd50['samples'][method], [2.5, 97.5])
    f_kw['nd50_y_position_on_plot'] = 0.08
    annotate_nd50_curve(x5, low_high[0], low_high[1], ax, 1, colors_nd50_methods[method], f_kw, nd50_limit=1/4 if d_c=='dilution' else 3.13)
    nd50['stats'][method] = (x5, low_high[0], low_high[1])

    method = 'Non-statistical'
    nd50_ns0 = [dilution for dilution, transduction in zip(x_data, y_curve.mean(axis=1))
                if transduction < 0.5]
    nd50_ns = nd50_ns0[-1] if nd50_ns0 else 1
    nd50['stats'][method] = (nd50_ns, nd50_ns, nd50_ns)
    f_kw['nd50_plot_capsize'] = 0
    f_kw['nd50_y_position_on_plot'] = 0.1
    annotate_nd50_curve(nd50_ns, nd50_ns, nd50_ns, ax, 1, colors_nd50_methods[method], f_kw, nd50_limit=1/4 if d_c=='dilution' else 3.13)
    f_kw['nd50_plot_capsize'] = 5
    return nd50


def nd50_mean_CI_converge(out_path=None, cv=0.1, n_repeats = 3, dilution_step = -1, seed=range(50), shift_range = 0):
    from scipy.stats import ttest_rel
    x_data = 2. ** np.arange(-2-shift_range, -9-shift_range, dilution_step)  # 7 dilutions
    nd50_hill = []
    nd50_lin = []
    nd50_ns = []  # non-statistical ND50
    fig, ax = plt.subplots(figsize=(5, plot_height))
    f_kw = {'nd50_y_position_on_plot': 0.05,
            'hill_error_style': [''],  #
            'mcmc_show_nd50': 0,
            }
    for s1 in seed:
        np.random.seed(s1)
        curve = generate_synthetic_data(x_data, 1, n_repeats, neutralization_type='strong', cv=cv)[0]
        m_CI, _ = mcmc_fit_plot(0.2, ax, 'blue', offset, x_data, curve, d_c='dilution', f_kw=f_kw)
        nd50_hill.append(m_CI)
        ##### Linear bootstrap
        bootstrap_nd50 = 2 ** bootstrap_nd50_lin_2_dilutions(np.log2(x_data), np.array(curve), avg_f=np.mean)
        x5 = bootstrap_nd50.mean()
        low_high = np.percentile(bootstrap_nd50, [2.5, 97.5])
        nd50_lin.append((x5, low_high[0], low_high[1]))

        # Non-statistical method: find the highest dilution with transduction < 50% of reference
        nd50_ns0 = [dilution for dilution, transduction in zip(x_data, curve.mean(axis=1)) if transduction < 0.5]
        nd50_ns.append(nd50_ns0[-1] if len(nd50_ns0) else 1)

    # Convert to numpy arrays for easier computation
    nd50_lin = np.array(nd50_lin)
    nd50_hill = np.array(nd50_hill)
    nd50_ns = np.array(nd50_ns)

    # Compute mean ND50 estimates
    mean_nd50_lin = nd50_lin[:, 0].mean()
    mean_nd50_hill = nd50_hill[:, 0].mean()
    mean_nd50_ns = nd50_ns.mean()

    # Compute credible interval widths
    ci_width_lin = nd50_lin[:, 2] - nd50_lin[:, 1]
    ci_width_hill = nd50_hill[:, 2] - nd50_hill[:, 1]

    # Perform paired t-test to compare means and CI widths
    t_stat_mean_lin_hill, p_value_mean_lin_hill = ttest_rel(nd50_lin[:, 0], nd50_hill[:, 0])
    t_stat_ci, p_value_ci = ttest_rel(ci_width_lin, ci_width_hill)
    t_stat_mean_ns_hill, p_value_mean_ns_hill = ttest_rel(nd50_ns, nd50_hill[:, 0])
    t_stat_mean_ns_lin, p_value_mean_ns_lin = ttest_rel(nd50_ns, nd50_lin[:, 0])

    add_fraction_ticks(plt.gca(), 'x', log=True, reciprocal=True)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=""))
    ax.set_ylabel('Transduction Efficiency (%)')
    # set_xscale adds xticks with log2 steps, set xticks explicitly to only include actual dilution points
    ax.set_xticks(x_data)
    ax.set_ylim(0,1.25)

    ci_of_ci_low = np.percentile(nd50_lin[:, 1], [2.5, 97.5])[0]
    ci_of_ci_high = np.percentile(nd50_lin[:, 2], [2.5, 97.5])[1]

    annotate_nd50_curve(mean_nd50_lin, ci_of_ci_low, ci_of_ci_high, ax, alpha, colors_nd50_methods['Linear-bootstrap'], f_kw)
    f_kw['nd50_y_position_on_plot'] = 0.08
    annotate_nd50_curve(mean_nd50_hill, nd50_hill[:, 1].min(), nd50_hill[:, 2].max(), ax, alpha, colors_nd50_methods['Hill-MCMC'], f_kw)

    ci_low, ci_high = np.percentile(nd50_ns, [2.5, 97.5])
    f_kw['nd50_y_position_on_plot'] = 0.1
    annotate_nd50_curve(mean_nd50_ns, ci_low, ci_high, ax, alpha, colors_nd50_methods['Non-statistical'], f_kw)

    # Print statistical results
    print(f"N repeats: {n_repeats} CV={cv}")
    print("Paired t-test for ND50 means:")
    print(f"Linear, Hill-MCMC: t = {t_stat_mean_lin_hill:.4f}, p = {p_value_mean_lin_hill:.4f}")
    print(f"Hill-MCMC vs Non-statistical: t = {t_stat_mean_ns_lin:.4f}, p = {p_value_mean_ns_lin:.4f}")
    print(f"Linear-bootstrap vs Non-statistical: t = {t_stat_mean_ns_lin:.4f}, p = {p_value_mean_ns_lin:.4f}")
    print(f"Paired t-test for CI widths Hill-MCMC vs Linear-boostrap: t = {t_stat_ci:.4f}, p = {p_value_ci:.4f}")
    legend0 = zip(['Synthetic data'], ['blue'], [':'])
    bbox0 = (0.725, 0.9)
    legend1 = zip(colors_nd50_methods.keys(), colors_nd50_methods.values(), ['--'] * 3)
    bbox1 = (0.733, 0.8) # horizontal, vertical position
    legend2 = zip([f'p = {p_value_mean_lin_hill:.4f} (means, Linear, Hill-MCMC)',
                   f'p = {p_value_ci:.4f} (credible intervals)',
                   ], ['grey']*2, ['--','-'])
    bbox2 = (0.525, 0.29)  # horizontal pos, vertical pos
    create_custom_legend([legend0, legend1, ], titles=['', 'ND50 estimation:', ], bbox=[bbox0, bbox1,], alpha=alpha)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path / \
                    Path(__file__).name.replace('.py', f"_nd50_lin_vs_hill_cv{cv}_{dilution_step}_nrep{n_repeats}_nseed{len(seed)}_shr{shift_range}.png"), dpi=plot_dpi)


def mcmc_fit_plot(alpha, ax, color, offset, x_data, y_curve, d_c, f_kw, refine=0, nd50_stats_dict=None):
    hill_trace, best_fit = fit_model(x_data, y_curve)
    bottom_samples, nd50_samples0, slope_samples = (hill_trace.posterior['bottom'].values.flatten(),
                                                    hill_trace.posterior['nd50'].values.flatten(),
                                                    hill_trace.posterior['slope'].values.flatten())
    top_samples = [1] * len(bottom_samples)
    # Visualize the fits for each curve
    keys = ['nd50_mean', 'nd50_lower_bound', 'nd50_upper_bound', 'dilution_at_y_0_5_samples']
    values = extrapolate_mcmc(hill_trace, 0.5)
    nd50_data = values[:3]
    nd50_samples = values[3]
    if ax is None:
        return nd50_data, nd50_samples

    if nd50_stats_dict is None:
        nd50_stats_dict = {}
    if 'plot_kw' not in nd50_stats_dict:
        nd50_stats_dict['plot_kw'] = {
            'title': '',
            'ylabel': 'Transduction Efficiency (%)',
            'xlabel': 'Serum Dilution',
            }
    nd50_stats_dict.update(dict(zip(keys, values)))
    nd50_stats_dict['hill_mean_style'] = {'linestyle': '-', 'alpha': alpha,}
    nd50_stats_dict['hill_x_offset'] = offset,
    nd50_stats_dict['plot_kw']['figures'] = f_kw

    if refine:
        x_data = np.linspace(x_data.min(), x_data.max(), len(x_data) * 10)

        # Evaluate Hill curve for all posterior samples
        posterior_evaluations = np.array([
            hill_curve(x_data, slope, nd50, 1, bottom)
            for slope, nd50, bottom in zip(slope_samples, nd50_samples0, bottom_samples)
        ])

        # Compute the mean of the posterior evaluations at each point
        y_curve = posterior_evaluations.mean(axis=0, keepdims=True).T

    visualize_mcmc_fit(x_data, y_curve, slope_samples, nd50_samples0, top_samples, bottom_samples, ax,
                       color=color, d_c=d_c, **nd50_stats_dict)

    # Plot original with confidence intervals
    lower_bound = np.percentile(y_curve, 2.5, axis=1)  # 2.5th percentile (lower bound)
    upper_bound = np.percentile(y_curve, 97.5, axis=1)  # 97.5th percentile (upper bound)
    meanc = np.mean(y_curve, axis=1)

    xoffs = 1.02  # allows inspection of CI for Hill fit and original data
    ax.plot(x_data*xoffs, meanc, color=color, alpha=alpha, linestyle=':', label='Mean of samples')
    # Manually add dotted error bars
    for xi, yi, yerr_lower, yerr_upper in zip(x_data, meanc, lower_bound, upper_bound):
        # Plot dotted vertical error bars
        plt.plot([xi*xoffs, xi*xoffs], [yerr_lower, yerr_upper], linestyle=':', color=color, alpha=alpha)

    plt.gca().set_ylim(0, upper_bound.max())
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    return nd50_data, nd50_samples


def adk9_demo(toml_path, out_path=None, nd50_thr_log=0, barplot_yaxis=1):
    from coretia.datahandling import import_plate, normalize, extras_from_meta
    from coretia.assay_quality import cv

    _colors = ['green', 'olive']
    out_path = toml_path.parent / 'output' if out_path is None else out_path
    out_path = out_path/toml_path.name.replace('.toml', f'nd50thr{nd50_thr_log}.png')

    df, extras, plates, d_c, category_pair = import_plate(toml_path)
    df1, all_single_dilution = normalize(df, 'conc_ng')

    # Filter for the specific sample
    sample_name = 'ADK9'
    filtered_df = df1[df1['sample'] == sample_name]

    intra_cv, inter_cv = cv(df, 'conc_ng')
    print(f'Intra-assay CV: {intra_cv}')

    # Omit conc_ng = 0 (no-antibody control)
    filtered_df = filtered_df[filtered_df['conc_ng'] != 0]

    # Extract x_data (concentrations) and y_curve (technical repeats)
    x_data = filtered_df['conc_ng'].unique()  # Get unique conc_ng values
    # Get the Lum values for each concentration in original order (3 technical repeats)
    y_curve = np.array([filtered_df[filtered_df['conc_ng'] == conc]['Lum'].head(3).values for conc in x_data])

    # Linear fit and bootstrap CI
    fkw0 = {'nd50_y_position_on_plot': 0.05,
            'mcmc_show_nd50': 1,
            'nd50_plot_marker_size': 0,
            'hill_error_style': ['hill_CI95_bars'],
            }
    op1 = out_path.with_name(out_path.stem + "_linear.png")

    linear_bootstrap_zoomed(x_data, y_curve, _colors[0], op1, fkw0, plot_height, dpi=plot_dpi)

    # Fit the model using x_data and y_curve
    fig1, ax = plt.subplots(figsize=(5, plot_height))
    f_kw = {'nd50_y_position_on_plot': 0.05,
            'mcmc_show_nd50': 0,
            'nd50_plot_marker_size': 0,
            'hill_error_style': ['hill_CI95_bars'],
    }

    nd50_stats_dict= {'plot_kw': {
        'title': '',
        'xlabel': 'Antibody Concentration [ng/mL]',
        'ylabel': 'Transduction Efficiency (%)'
    }}
    nd50 = nd50_3_methods(ax, 'green', f_kw, x_data, y_curve, d_c='conc_ng', nd50_stats_dict=nd50_stats_dict)
    legends = []
    titles = []
    #bbox = [(0.68, 0.79), (0.72, 0.68)]  # horizontal, vertical pos
    bbox = [(0.715, 0.87), (0.733, 0.76)]
    ccolor = ['green'] * 2
    legends.append(zip(["Raw Samples", "Hill-fit"],
                       ccolor,
                       [':', '-']
                       ))
    legends.append(zip(colors_nd50_methods.keys(), colors_nd50_methods.values(), ['--'] * 3))
    titles.extend(['', 'ND50 estimation:'])
    create_custom_legend(legends, titles=titles, bbox=bbox, alpha=alpha)
    plt.tight_layout()
    plt.gca().figure.savefig(out_path, dpi=plot_dpi)
    plt.close()

    #### Barplot with Hill-MCMC and linear bootstrap
    labels = ['Non-statistical', 'Linear-bootstrap', 'Hill-MCMC']
    plot_kw = {'figures': {'mcmc_nd50_pair_with_next': 1,
                           'mcmc_barplot_significance': 'star',
                           'label_rotation': 0, 'label_ha': 'center',
                           'barplotsize': (2, plot_height), 'mcmc_barplot_significance_positions': 'max',
                           'nd50_labels': [f'Thr = {nd50_thr_log}\nlog2 units'],
                           'barplotylim': extras['figures'].get('barplotylim'),
                           }
               }
    plot_kw['figures']['nd50_labels'] = labels
    plot_kw['figures']['label_rotation'] = 40
    plot_kw['figures']['label_ha'] = 'right'
    plot_kw['figures']['barplot_text_gap'] = extras['figures'].get('barplot_text_gap')
    plot_kw['figures']['barplot_toffset'] = extras['figures'].get('barplot_toffset')

    plot_kw['figures']['barplot_yaxis'] = barplot_yaxis
    plot_kw["xlabel"] = 'Antibody Concentration [ng/mL]'  # Barplot y label uses x label of raw dataset (dilution or conc_ng)

    nd50['samples']['Non-statistical'] = None
    # Recreate dict with key ordering:
    for k0 in ['stats', 'samples']:
        nd50[k0] = {k1: nd50[k0][k1] for k1 in colors_nd50_methods.keys()}
    fix = nd50_barplot_mcmc(['Non-statistical', 'Linear-bootstrap', 'Hill-MCMC'], nd50['stats'], nd50['samples'], 'conc_ng',
                            colors=list(colors_nd50_methods.values()),
                            plot_kw=plot_kw, nd50_thr_log=nd50_thr_log)
    title_within_axes(f"θ = {nd50_thr_log}\nlog2 units", fontsize=8)
    plt.tight_layout()
    if out_path is not None:
        op2 = out_path.with_name(out_path.stem + "_linear.png")
        fix.savefig(op2, dpi=plot_dpi)

    return


if __name__ == '__main__':
    # Set the seed for NumPy for reproducible results
    fire.Fire()