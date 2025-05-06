"""
Process toml file provided by command line parameter.
Command line steps:
1. cd to the folder where the package is located (C:\pycharmprojects\coretia)
2. uv run python process.py recursive mytoml.toml

These steps will launch script that recursively processes directory with toml files or single toml file.

Note: when using command line arguments, use one backslash, not two as in the docstring here.

Additional options:
If you want to visualize the plate layout, you can append:
--show_plate
name the method you want to show, e.g. 'raw', 'transduction', 'transduction0'
to the end of the poetry.... command.

Example to show population plots:
uv run python process.py recursive C:\\Users\\measurement\\PycharmProjects\\coretia\\coretia\\data\\20230720 process_plate

"""
out_base = None
import traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tomllib
import coretia.datahandling as datahandling
import coretia.plot as plot
from coretia import outpath_from_toml
import fire

estimators = ['Linear-bootstrap', 'Hill-MCMC']
default_estimator = 'Hill-MCMC'  # 'Linear-bootstrap' 'Hill-MCMC'

# Set here which functions can be used to process multiple TOML files in a directory

def recursive(start_path: str, analysis_name: str = 'process', extension: str = 'toml', contains: str = None, **kwargs):
    '''
    Parameters
    ----------
    analysis_name: the name of the method you want to use from this module, e.g. process_plate
    start_path: e.g. C:\\Users\\measurement\\PycharmProjects\\coretia\\coretia\\data
                Can end with * to match files with a prefix, e.g. /User/project/files*
    extension: e.g. 'toml'
    kwargs: parameters to be passed to the function called

    Returns
    -------
    '''
    functions = globals()  # list of defined methods that 'analysis' can call, defaults to process
    if analysis_name not in functions:
        raise KeyError(f"Selected function {analysis_name} not found. Available: {functions.keys()}")

    # Handle wildcard at the end of path
    if start_path.endswith('*'):
        # Get the directory and filename prefix
        prefix_path = Path(start_path[:-1])
        parent_dir = prefix_path.parent
        prefix = prefix_path.name

        if parent_dir.is_dir():
            # Find files that start with the prefix and have the right extension
            files = list(parent_dir.glob(f'{prefix}*.{extension}'))
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No {extension} files matching pattern '{start_path}.{extension}' found.")
        else:
            raise FileNotFoundError(f"Directory {parent_dir} does not exist.")
    elif Path(start_path).is_dir():
        files = list(Path(start_path).glob(f'**/*.{extension}'))
        if len(files) == 0:
            raise FileNotFoundError(
                f"Folder {start_path} does not contain any {extension} files, or is it a file, not a folder?")
    else:
        files = [Path(start_path)]

    processed = []

    for f1 in files:
        if any([p1 in f1.name for p1 in processed]):  # processed can be just a filename retrieved from [meta.concat] in aggregator TOML files
            continue
        if contains is not None and contains not in f1.name:
            continue
        print(f"Processing {f1}")
        try:
            # one TOML file can refer to multiple, those are returned here and will not be processed again
            with open(f1, 'rb') as tf1:
                tcontent = tomllib.load(tf1)
            if 'meta' in tcontent and 'concat' in tcontent['meta']:
                processed.extend(tcontent['meta']['concat'].values())

            functions[analysis_name](f1, **kwargs)

        except Exception as e1:
            traceback.print_exc()


def process(toml_path, show_plate=False, sigmoid=False):
    """
    Entry point to analyze and plot experiments in multi-plate TOML files.
    """
    if show_plate:
        import wellmap
        fig = wellmap.show(Path(toml_path))
        fig.savefig(outpath_from_toml(Path(toml_path), remap_to = out_base).with_suffix('.png'), dpi=300)
        return
    out_txt_path = outpath_from_toml(Path(toml_path), remap_to = out_base).with_suffix('.txt')
    with redirect_print_to_file_and_stdout(out_txt_path):
        # Load and preprocess data
        df, extras, plates, d_c, category_pair = datahandling.import_plate(toml_path)

        from coretia.assay_quality import cv
        intra_cv, inter_cv = cv(df, d_c)

        if 'dilution' in df:   # convert from 4, 8... to 1/4, 1/8...
            df['dilution'] = 1 / df['dilution']

        # Extract plot configurations
        plot_kw = {pfeature: extras.get(pfeature) for pfeature in ['xlabel', 'ylabel', 'title', 'figures', 'plot', 'dpi', 'fontsize',
                                                                   'nd50_thr_log', 'nd50_estimator', 'minor_labels']
                    if pfeature in extras}

        # Don't write images if TOML file has no instructions (not intended for single plot)
        out_path = None if len(plot_kw)==0 else outpath_from_toml(Path(toml_path), remap_to = out_base).with_suffix('.pdf')

        # Plot all curves, raw and normalized, return nd50 mean, CI low, CI high (for dilution curves)
        all_single_dilution, nd50_mlh_bootstrap = plot.plot_all(df, out_path, d_c, plot_kw, category_pair, plates=plates)

        # Validate and process grouping conditions
        if not ('dilution' in d_c or 'conc_ng' in d_c):
            if d_c in ['fbs', 'serum', 'moi']:
                return False  # Exit here, no other analysis will be ran
            raise RuntimeError("TOML file lacks required 'dilution' or 'conc_ng'.")

        # Perform Bayesian modeling if full curves are processed, not single dilution points
        if not all_single_dilution and plot_kw.get('nd50_estimator', default_estimator) == 'Hill-MCMC':
            nd50_mlh_bayes = perform_bayesian_modeling(toml_path, out_path, category_pair, plot_kw, require_sigmoid = sigmoid)
        else:
            return None

        # Assign xls path as key to ND50 calculations, this can identify same plates when performing ND50 adjustment
        nd50_out = nd50_mlh_bootstrap if plot_kw.get('nd50_estimator', default_estimator) == 'bootstrap' else nd50_mlh_bayes

        nd50_out_with_path, path_mapping = replace_nd50_key_xlsname(df, nd50_out)
        nd50_out = {k1: tuple(v1) + (intra_cv[k1],) for k1, v1 in nd50_out.items()}

        # nd50_out_with_path: for inter-plate ND50 reference adjustment
        # plot_kw.get('nd50_estimator', default_estimator): 'bootstrap' or 'Hill-MCMC'
        # nd50_out: for analyzing ND50-cv relationships in assay_quality.py
        return nd50_out_with_path, plot_kw.get('nd50_estimator', default_estimator), nd50_out


def replace_nd50_key_xlsname(df, nd50_out):
    # Drop duplicates to ensure unique combinations of sample, condition, and path
    unique_df = df[['sample', 'condition', 'path', 'plate']].drop_duplicates()
    # Check assumption that df['condition'] and nd50_out.keys() map to each other:
    nd50_out_maps_to_df_condition = [1 if k1 in unique_df['condition'].tolist() else 0 for k1 in nd50_out.keys()]
    nd50_out_maps_to_df_plate = [1 if k1 in unique_df['plate'].tolist() else 0 for k1 in nd50_out.keys()]
    if all(nd50_out_maps_to_df_condition):
        path_mapping = {f"{row['condition']}": row['path'].name for _, row in unique_df.iterrows()}
    elif all(nd50_out_maps_to_df_plate):
        # Use plate names overridden in aggregator TOML:
        path_mapping = {f"{row['plate']}": row['path'].name for _, row in unique_df.iterrows()}
    else:
        raise RuntimeError(f"Inconsistent sample names, check aggregator TOML and source TOMLs: {nd50_out.keys} <-> {unique_df['condition']}")
        path_mapping = {f"{row['sample']} {row['condition']}": row['path'].name for _, row in unique_df.iterrows()}
    # Replace the keys in nd50_out with the corresponding path
    nd50_out_with_path = {path_mapping[key]: value for key, value in nd50_out.items()}
    return nd50_out_with_path, path_mapping


def perform_bayesian_modeling(toml_path, out_path, category_pair, plot_kw, nd50_reference=None, require_sigmoid=False):
    """
    Perform Bayesian modeling on the data if needed.
    """
    x1, y1, data, df, extras, plates, d_c = datahandling.import_multiplate(toml_path)

    bayes_out_path = out_path.with_stem(out_path.stem + "_bayes") if out_path is not None else None
    if plot_kw.get('figures', {}).get('nd50_labels_override', 0) == 0:
        subjects = feasible_pairs(df, d_c)
    else:
        override_labels = plot_kw.get('figures', {}).get('nd50_labels')
        if override_labels is None:
            raise RuntimeError(f"TOML file 'nd50_labels_override' set to 1 but 'nd50_labels' is not defined.")
        subjects = override_labels

    nd50_mlh = curve_pairs_pvals(x1, y1, plates, d_c, category_pair, plot_kw=plot_kw, out_path=bayes_out_path,
                                 title=extras['title'] if 'title' in extras else '', subjects=subjects, df=df, nd50_reference=nd50_reference)

    if require_sigmoid:
        is_sigmoid = {pk: datahandling.curve_is_full_sigmoid(y2, 0.5, 0.5, debug=1) for y2, pk in zip(y1, plates)}
        nd50_mlh = {k1 : v1 for k1, v1 in nd50_mlh.items() if is_sigmoid[k1]}
    return nd50_mlh


def feasible_pairs(df2, d_c):
    # Check if any sample has data for both categories (conditions)
    # Group the data by sample_id and condition, and count the number of conc_ng values
    grouped = df2.groupby(['sample', 'condition']).agg({d_c: 'count'}).reset_index()
    # Filter out the groups where each condition has at least 5 dilution/conc_ng/serum/FBS values
    valid_groups = grouped[grouped[d_c] >= 5]
    # Now pivot the valid_groups table to check if both conditions are present for each sample_id
    pivot = valid_groups.pivot(index='sample', columns='condition', values=d_c)
    # Check for sample_ids where both conditions (DMEM and FBS) are present
    feasible_pairs = pivot.dropna().index.tolist()
    return feasible_pairs


def estimate_log_base(dilution_series, log_options = (2, 10)):
    # Determine the log base from the dilution series
    ratio = dilution_series[0] / dilution_series[1]  # Compute the first ratio
    ratio_match = [np.abs(np.log(ratio) / np.log(lo1)) for lo1 in log_options]  # Logarithmic base (log2, log10, etc.)

    if not any([np.isclose(lb1, 1) for lb1 in ratio_match]):
        raise ValueError("Dilution series does not follow a clear logarithmic pattern.")
    else:
        # Choose the log base that is closest to an integer
        best_base = next(lb1 for r1, lb1  in zip(ratio_match, log_options) if np.isclose(r1, 1))
        return best_base


def curve_pairs_pvals(x1, y1, plates, d_c, categories, plot_kw=None, out_path=None, title=None, subjects=None, df=None, nd50_reference=None):
    """
    Fits Hill curve using Markov Chain Monte Carlo method and compares ND50 estimates

    Parameters
    ----------
    x1: list of dilutions or conc_ng vectors, for 1 curve: [array([0.005, 0.01 , 0.02 , 0.039, 0.078, 0.156, 0.313])]
    y1: list of normalized luminescence values, for 1 curve: [array([[0.9075276 , 0.88924221, 0.95749633],
       [0.76846348, 0.74382647, 0.81783267],
       [0.62616102, 0.59643537, 0.62514819],
       [0.34994426, 0.34637147, 0.34331394],
       [0.14809669, 0.12420877, 0.12976508],
       [0.0238485 , 0.02209473, 0.02856055],
       [0.036965 , 0.00179591, 0.0064767 ]])]
    plates: will be used as plot labels, e.g. ['20240223 FBS']
    d_c: 'dilution' or 'conc_ng'
    categories: if pairs of curves are plotted
    plot_kw: {'xlabel': 'Antibody Concentration [ng/well]', 'ylabel': 'Transduction Efficiency (%)', 'title': 'ADK9 antibody against AAV9', 'figures': {'allcurves': 'SFig 3a', 'loghillplots': 'Fig 4a', 'mcmc_show_nd50': 1}, 'plot': {'groupby': ['sample']}}
    out_path: PosixPath('/Users/hd/PycharmProjects/coretia/coretia/data/paper/Fig1_MCMC/output/adk9_fbs_bayes.pdf')

    Returns
    -------

    """
    from coretia.nl_model import Curve, Comparer
    from coretia.plot import generate_even_colors

    plot_kw = plot_kw or {}

    # Create curves from x1, y1, and plates
    curves = [Curve(x1[i1], y1[i1], name=plates[i1], d_c=d_c) for i1 in range(len(y1))]

    log_bases = [estimate_log_base(x1[i1] if d_c=='dilution' else x1[i1]) for i1 in range(len(y1))]
    if not all([lb1 == log_bases[0] for lb1 in log_bases]):
        raise RuntimeError(f"Not all curves used the same dilution step, got {log_bases}")
    else:
        log_base = log_bases[0]

    # Use None for categories to get different color for each curve
    colors, linestyles = plot.get_colors_for_curves(len(y1), None, plates, plot_kw)

    # Initialize comparer to compare curves
    comparer = Comparer(curves)

    # Plot settings
    n = len(comparer.i_pairs) + 2  # Bar plot + initial guess plot

    # Perform Hill fitting for all curves
    [c1.hill_fit() for c1 in curves]

    # Plot Hill fits if the flag is set
    if 'plot_hill_fit' in plot_kw:
        plot_hill_fits(curves, colors, out_path)

    # Perform Bayesian comparison of all curves
    comparer.compare_all_bayes(categories=categories, out_path=out_path, nd50_thr_log=plot_kw.get('nd50_thr_log', 0.25))

    # Plot MCMC fits
    plot_mcmc_fits(x1, d_c, comparer, curves, colors, plot_kw, out_path)

    # Plot bar chart if there are multiple curves
    if len(curves) > 1:
        colors, linestyles = plot.get_colors_for_curves(len(y1), categories, plates, plot_kw, color_key = 'nd50_colors')

        plot_nd50_barplot_mcmc(subjects, comparer.nd50_bt, comparer.nd50_samples, d_c, colors, linestyles, out_path,
                               plot_kw, categories=categories, log_base=log_base, nd50_thr_log=plot_kw.get('nd50_thr_log', 0.3))

    return comparer.nd50_bt


# Helper function to plot Hill fits
def plot_hill_fits(curves, colors, out_path):
    fix, ax = plt.subplots()
    [c1.plot(ax, color=colors[ci]) for ci, c1 in enumerate(curves)]
    ax.figure.legend(loc='outside right upper')
    ax.set_xscale('log', base=2)

    if out_path is not None:
        fix.tight_layout()
        fix.savefig(out_path.with_stem(out_path.stem + "_allplots").with_suffix('.png'), dpi=600)
    else:
        plt.show(block=True)
    plt.close()


# Helper function to plot MCMC fits
def plot_mcmc_fits(x1, d_c, comparer, curves, colors, plot_kw, out_path=None):
    from coretia.bayesian import mcmc_non_neutralizing

    figsize = plot_kw.get('figures',{}).get('mcmc_curves_figsize', (5,6))
    fix, ax = plt.subplots(figsize=figsize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    pkw = plot_kw.copy()
    pkw['title'] = ''  # title will be added later
    [c1.visualize_mcmc_fit(ax, color=colors[ci], plot_kw=plot_kw) for ci, c1 in enumerate(curves)]

    for ci, c1 in enumerate(curves):
        eval_at = 1 / min(x1[ci]) if d_c == 'dilution' else max(x1[ci])  # min(4) or max cc = 1.25 ng
        nn = mcmc_non_neutralizing(c1.trace, eval_at)

        # Plot ND50 line only for neutralizing curves
        if not nn:  # neutralizing
            nx = [comparer.nd50_bt[curves[ci].name][0]] * 2
            ny = [0.5, 0.5]
            ax.plot(nx, ny, color=colors[ci], linestyle=':')
        else:  # non-neutralizing ND50 is set to 1
            # nd50 median, lower, higher CI
            if comparer.nd50_bt[curves[ci].name][0] > 1:
                comparer.nd50_bt[curves[ci].name] = [min(1, v1) for v1 in comparer.nd50_bt[curves[ci].name]]

    legend_on = plot_kw.get('figures',{}).get('mcmc_curveslegend',['best'])  # default is show legend

    if legend_on:
        ncols = np.ceil(len(curves) / 12)
        legend_str = [lp1 for lp1 in plot.legend_locations if lp1 in legend_on[0]]
        if legend_str:
            if legend_str[0] == 'outside':
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, ncol=ncols)
            else:
                plt.legend(frameon=False, loc=legend_str[0], ncol=ncols)
    if 'title' in legend_on:
        plot.title_within_axes(plot_kw.get("title", 'AAV coretia Data'), **plot_kw)

    ax.set_ylim(0,ax.get_ylim()[1])

    if out_path is not None:
        fix.tight_layout()
        op1 = plot.generate_output_name(out_path, "loghillplots", plot_kw)
        fix.savefig(op1.with_suffix('.png'), dpi=plot_kw.get('dpi',plot.plot_dpi))
    else:
        plt.show()
    plt.close()


# Helper function to plot the ND50 barplot
def plot_nd50_barplot_mcmc(subjects, nd50_data, nd50_samples, d_c, colors, linestyles, out_path, plot_kw=None, hatch=None,
                           categories = None, log_base=2, nd50_thr_log=0.5):
    from coretia.bootstrap import nd50_barplot_mcmc
    hatch = ['-' if h1 == '--' else None for h1 in linestyles]
    fix = nd50_barplot_mcmc(subjects, nd50_data, nd50_samples, d_c, colors=colors, hatch = hatch, plot_kw=plot_kw,
                            categories=categories, log_base=log_base, nd50_thr_log=nd50_thr_log)

    if out_path is not None:
        op1 = plot.generate_output_name(out_path, "barplot", plot_kw)
        fix.savefig(op1.with_suffix('.png'), dpi=plot_kw.get('dpi',plot.plot_dpi))
    else:
        plt.show(block=True)
    plt.close()


## Print statements show in stdout and file:
from contextlib import contextmanager
import sys


@contextmanager
def redirect_print_to_file_and_stdout(file_path):
    # Open the file for writing
    with open(file_path, 'w', encoding='utf-8') as f:
        # Save the original stdout
        original_stdout = sys.stdout
        try:
            # Redirect stdout to both file and console
            sys.stdout = Tee(original_stdout, f)
            yield  # This is where the code that uses print() will run
        finally:
            # Restore original stdout
            sys.stdout = original_stdout


class Tee:
    """Class to redirect print output to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()  # Ensure the message is flushed to the stream

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == '__main__':
    fire.Fire()