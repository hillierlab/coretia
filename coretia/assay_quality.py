""" Process data from uninjected or presumably AAV-free subjects """
import os

from coretia import datapath as datapath
import fire
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from pathlib import Path

compare_pairs = \
    [('sample', (datapath/'other/adk9_moi1000_20240810.toml', datapath/'firefly/adk9_moi1000_20240810.toml'), 0),
     ('moi', (datapath/'other/adk9_moi1000_20240810.toml', datapath/'other/adk9_moi100_20240810.toml'), 0)
     ]

species_strings = {'Human': ['TRX'], 'Macaque': ['Lala', 'Rafiki', 'Sti', 'Pikachu', 'Jango'], 'Cat': ['Cirmi', 'Lulu'],
                   'Rat': ['Rat'], 'ADK': ['ADK']}


def cv(df, d_c):
    """
        Calculate intra-assay and inter-assay coefficient of variation (CV).

        The function first normalizes the data using the provided free variable column ('dilution', 'cong_ng') by calling the
        'normalize' function. It then computes:
          - Intra-assay CV: Calculated for each plate and dilution as (std/mean)*100.
          - Inter-assay CV: Calculated for each dilution across plates, then averaged across dilutions.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing raw assay measurements.
        d_c : str
            Column name used for normalization (e.g., 'conc_ng', 'dilution', etc.).

        Returns
        -------
        'intra_assay_cv': DataFrame containing intra-assay CV per plate and dilution.
        'inter_assay_cv': Scalar average inter-assay CV.
        """
    from coretia.datahandling import normalize
    if d_c not in ['dilution', 'conc_ng']:
        return None, None
    df1, all_single_dilution = normalize(df, d_c)
    intra_assay_cv = df1.groupby(['plate', d_c]).agg(
        Lum_mean=('Lum', 'mean'),
        Lum_std=('Lum', 'std')
    )
    intra_assay_cv['CV_intra'] = (intra_assay_cv[('Lum_std')] / np.abs(intra_assay_cv[('Lum_mean')])) * 100

    inter_assay_agg = intra_assay_cv.groupby([d_c]).agg(
        mean_of_means=('Lum_mean', 'mean'),
        std_of_means=('Lum_mean', 'std')
    )

    # Calculate the inter-assay CV as (std of means / mean of means) * 100
    inter_assay_agg['CV_inter'] = (inter_assay_agg['std_of_means'] / inter_assay_agg['mean_of_means']) * 100
    inter_assay_cv = inter_assay_agg['CV_inter'].mean()

    # Display the result
    print(f"Intra-assay CV = {intra_assay_cv['CV_intra'].mean():1.1f}")
    print(f"Inter-assay CV = {inter_assay_cv:1.1f} !! This only makes sense when exact same sample is tested across plates, eg. TRX-045_variability.toml")
    if any(intra_assay_cv['CV_intra'].groupby(level='plate').mean() <= 0):
        print("Negative intra-assay CV, likely due to ~0 but negative values at highly neutralizing dilutions")
    return intra_assay_cv['CV_intra'].groupby(level='plate').mean(), inter_assay_cv


def parse_nd50_db(file_path, use_species = 'all', ignore_keywords = None, verbose = 0):
    """Parse database and return all data with paired samples marked.

    Returns:
        tuple: (all_ci_data, paired_ci_data) where:
            - all_ci_data: Dict {'bootstrap': [], 'Hill-MCMC': []} with ALL samples
            - paired_ci_data: List of (bootstrap_ci, hill_ci, cv) tuples
    """
    file_path = Path(file_path)
    out_path = file_path.parent / (file_path.stem+f' {use_species}')
    out_path.mkdir(parents=True, exist_ok=True)
    import shelve
    import re

    key_pattern = re.compile(r"^(.*?)[_ ](bootstrap|Hill-MCMC)$", re.IGNORECASE)

    all_ci_data = defaultdict(list)
    paired_data = []
    sample_store = defaultdict(dict)
    species_count = dict.fromkeys(species_strings.keys(),0)
    with shelve.open(file_path.with_suffix('')) as result_db:  # shelve file name must be without the .db suffix
        for key in result_db:
            match = key_pattern.match(key)
            if not match:
                continue

            sample_id, method = match.groups()
            method = 'Hill-MCMC' if 'hill' in method.lower() else method.lower()
            cv = result_db[key]

            # Store in both all_data and sample_store
            for sk1, mlh in cv.items():
                if ignore_keywords is not None and any([kw in sk1 for kw in ignore_keywords]):
                    continue
                if any(np.isnan(x) for x in mlh) or mlh[0] >= 0.25 or mlh[1] == mlh[0]:
                    continue
                species = [k1 for k1, st_list in species_strings.items() if any([s1.lower() in sk1.lower() or s1.lower() in sample_id.lower() for s1 in st_list])]
                if len(species) == 0:
                    raise RuntimeError(f"No species could be identified for {sk1} {sample_id}.")
                elif len(species) == 1:
                    if use_species == 'all' or species[0] in use_species:
                        species_count[species[0]] += 1
                    else:
                        if verbose:
                            print(f"Species {species[0]} excluded, only {use_species} included.")
                        continue
                low, high = mlh[1], mlh[2]
                ci_diff = np.log2(high) - np.log2(low)
                all_ci_data[method].append(ci_diff)

                # Build sample store for pairing
                sample_key = sample_id + sk1
                if sample_key not in sample_store:
                    sample_store[sample_key] = {}
                if method not in sample_store[sample_key]:
                    sample_store[sample_key][method] = []
                new_ci_cv = (ci_diff, mlh[3])
                # Samples can be used in multiple aggregator TOMLs, check if CI, CV pair is already in list to prevent duplication
                if not any([new_ci_cv == collected for collected in sample_store[sample_key][method]]):
                    sample_store[sample_key][method].append(new_ci_cv)  # CI_diff, CV
    # Create paired dataset from sample_store
    for sample_id, methods in sample_store.items():
        if 'bootstrap' in methods and 'Hill-MCMC' in methods:
            for b_ci, h_ci in zip(methods['bootstrap'], methods['Hill-MCMC']):
                #assert b_ci[1] == h_ci[1], f"CV not identical for {sample_id}, bootstrap {b_ci}, hill {h_ci}"  # CV should be identical for same data
                paired_data.append((b_ci[0], h_ci[0], b_ci[1]))  # Using last cv value
    print(species_count)
    return dict(all_ci_data), paired_data, out_path, species_count


def plot_fixed_bin_histograms(all_ci_data, out_path, species_count, medians):
    """Plot histograms using ALL samples."""
    from coretia.data.Fig2_MCMC.basic_comparing_neutralization_levels import colors_nd50_methods

    bin_edges = np.arange(0, 1.05, 0.05)  # /distance between ND50 and CI
    medians = [m1 for m1 in medians]  # half widths will be plotted
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Center of each bin
    bar_width = 0.01  # Width of each bar (adjust as needed)

    fig, ax = plt.subplots(figsize=(5, 6))
    colors = {'bootstrap': colors_nd50_methods['Linear-bootstrap'], 'Hill-MCMC': colors_nd50_methods['Hill-MCMC']}
    labels = {'bootstrap': 'Linear-bootstrap', 'Hill-MCMC': 'Hill-MCMC'}
    cutoffs = {}  # To store cutoff values for legend

    for i, (method, color) in enumerate(colors.items()):
        values = np.array(all_ci_data.get(method, []))
        counts, _ = np.histogram(values, bins=bin_edges)

        # Plot histogram
        offset = bar_width * (i - 0.5)  # Shift bars for each method
        ax.bar(bin_centers + offset, counts, width=bar_width,
               color=color, alpha=0.4, label=method)

        # Calculate cutoff if valid data exists
        if counts[:-1].sum() > 0:
            cumulative = np.cumsum(counts[:-1]) / counts[:-1].sum()
            cutoff_bin = np.argmax(cumulative >= 0.9) + 1  # +1 for bin edge index
            cutoff = bin_edges[cutoff_bin]
            cutoffs[method] = cutoff
            # Vertical line at 90% cutoff, stop at 10 to leave space for legend
            ax.plot([cutoff, cutoff], [0, 10], color=color, linestyle='--', linewidth=1.5, alpha=1)

            ax.plot([medians[i]]*2, [0, 0.2], color=color, linewidth=1.5)

    # Create combined legend
    legend_elements = ([
                          plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=labels[method])
                          for method, color in colors.items()
                      ]+[
                          plt.Line2D([0], [0], color=colors[method], linestyle='--', linewidth=1.5,
                                     label=f'90% cutoff: {cutoff:.2f}')
                          for method, cutoff in cutoffs.items()
                      ])


    ax.set(xlim=(-0.02, 0.8), xlabel='Credible Interval Width (log2 unit)', ylabel='Frequency',)
           #title=f'CI Width Distribution (All Samples, N={species_count})')
    #ax.set_xticks(np.append(np.arange(0, 1.05, 0.2), 1.05))
    #ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.legend(handles=legend_elements, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        fig.savefig((out_path / 'boostrap-hill_histogram').with_suffix('.png'), dpi=300)


def analyze_paired_samples(paired_data):
    """Perform statistical analysis on strictly paired samples."""
    if not paired_data:
        print("No paired samples found")
        return

    b_ci, h_ci, cv = zip(*paired_data)
    diffs = [b - h for b, h in zip(b_ci, h_ci)]
    medians = [np.median(b_ci), np.median(h_ci)]
    print(f"Paired samples: {len(paired_data)}")
    print(f"Bootstrap median CI: {medians[0]:.2f}")
    print(f"Hill-MCMC median CI: {medians[1]:.2f}")

    # Wilcoxon signed-rank test
    res = stats.wilcoxon(diffs)
    print(f"\nWilcoxon signed-rank test:")
    print(f"  Statistic: {res.statistic:.0f}")
    print(f"  p-value: {res.pvalue:.4f}")

    # Proportion where Hill-MCMC is better
    better_hill = sum(h < b for b, h in zip(b_ci, h_ci)) / len(paired_data)
    print(f"\nHill-MCMC produced narrower CI in {better_hill:.1%} of cases")
    return medians


def ci_cv_figure(db_path):
    from coretia.data import cache_data
    def run(s1):
        cache_path = Path(db_path).parent / f"nd50_{s1}_{Path(db_path).stem}.pkl"
        all_ci_data, paired_data, species_count  = cache_data(cache_path)
        medians = analyze_paired_samples(paired_data)
        os.makedirs(Path(db_path).parent/'output', exist_ok=True)
        plot_fixed_bin_histograms(all_ci_data, Path(db_path).parent/'output', species_count, medians)

    my_species = ['Human', 'Macaque']
    run(my_species)
    return


if __name__ == "__main__":
    fire.Fire()



