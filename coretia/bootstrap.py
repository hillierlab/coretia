from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import re
from itertools import combinations

from coretia.plot import darken_color, add_fraction_ticks, title_within_axes, plot_dpi

colors = ['orange', 'blue']


def f_pval(p_value, exponent=True):
  if 0.01 <= p_value <= 1:
      return f"$p={p_value:.2f}$"
  elif 0 < p_value < 0.01:
      if exponent:
          exponent = int(-np.log10(p_value))
          return f"$p=10^{{-{exponent}}}$"
      else:
          return "$p<0.01$"
  else:
      raise ValueError(f"p-value should be <1, got {p_value}")  # or any other indicator for values outside the specified range


# Visualization function to show data, fitted models, and confidence intervals
def visualize_mcmc_fit(x_data, y_curve, slope_samples, nd50_samples, top_samples, bottom_samples, ax=None, color=None,
                       name='', d_c='dilution', **kwargs):
    """
    Visualize the Hill curve fit with observed data and 95% credible intervals.

    Parameters:
    - x_data: array of dilution values
    - y_curve: 2D array of responses (7 dilutions x 3 replicates)
    - trace: posterior samples from the Hill curve model
    - trace_index: which curve to plot (if multiple curves were estimated simultaneously)
    """
    from coretia.nl_model import hill_curve

    plot_kw = kwargs.get('plot_kw', {})
    fkw = plot_kw.get('figures',{})
    ax = plt.gca() if ax is None else ax

    # Compute Hill curves for each set of posterior samples
    hill_curves = np.zeros((len(bottom_samples), len(x_data)))
    for i in range(len(bottom_samples)):
        hill_curves[i] = hill_curve(x_data, slope_samples[i], nd50_samples[i], top_samples[i], bottom_samples[i])

    # Compute posterior mean for the Hill curve
    hill_mean = np.nanmean(hill_curves, axis=0)

    # Compute 95% credible intervals
    lower_hill = np.nanpercentile(hill_curves, 2.5, axis=0)
    upper_hill = np.nanpercentile(hill_curves, 97.5, axis=0)

    # Plot Hill fit (mean of posterior)
    x_offset_factor = plot_kw.get('hill_x_offset', 0)
    if x_offset_factor:
        x_data = x_data * 2**x_offset_factor
    linestyle = plot_kw.get('hill_mean_style', {}).get('linestyle')
    alpha = plot_kw.get('hill_mean_style', {}).get('alpha', 1)

    if fkw.get('hill_error_style') is None and 'hill_CI95_shading' not in fkw.get('hill_error_style', ['hill_CI95_shading']):
        ax.plot(x_data, hill_mean, color=color, linestyle=linestyle, alpha=alpha, label=f"{name}")
    if 'orig_std' in fkw.get('hill_error_style', []):
        # Plot data points
        ax.errorbar(x_data, y_curve.mean(axis=1), color = color, yerr=y_curve.std(axis=1), fmt='o', alpha=alpha)#, label=f"{name}, observed")
    if 'hill_CI95_shading' in fkw.get('hill_error_style', ['hill_CI95_shading']):  # default is ON
        # Add 95% credible interval
        ax.plot(x_data, hill_mean, color=color, linestyle=linestyle, alpha=alpha, label=f"{name}")
        ax.fill_between(x_data, lower_hill, upper_hill, color=color, alpha=alpha/4)#, label='Hill Fit (95% CI)')
    if 'hill_CI95_bars' in fkw.get('hill_error_style', []):
        err_obj = ax.errorbar(x_data, hill_mean, color = color, yerr=[hill_mean-lower_hill, upper_hill-hill_mean], fmt='',
                    alpha=alpha, linestyle=linestyle)
        # Update the label to only use the line part for the legend (and not the vertical error barlet)
        err_obj[0].set_label(f"{name}")  # Only the line part will appear in the legend

    if fkw.get('mcmc_show_nd50',1) != 0:
        # Plot ND50 credible interval
        nd50_s = [kwargs.get(nd50k) for nd50k in ['nd50_mean', 'nd50_upper_bound', 'nd50_lower_bound']]
        nd50_in_range = True
        if plot_kw.get('figures',{}).get('mcmc_show_nd50',1) != 0:
            # If TOML parameter requires not showing ND50 above smallest dilution (1/4):
            nd50_in_range = nd50_s[0] <= sorted(x_data)[-1]

        if nd50_in_range and all(nd50_s):  # nd50_stats pre-computed
            nd50_mean, nd50_upper_bound, nd50_lower_bound = nd50_s
            annotate_nd50_curve(nd50_mean, nd50_lower_bound, nd50_upper_bound, ax, alpha, color, fkw, nd50_limit=4 if d_c=='conc_ng' else 1/4)

    if d_c == 'dilution':
        add_fraction_ticks(plt.gca(), 'x', log=True, reciprocal=True)
    elif d_c == 'conc_ng':
        tick_positions = x_data
        tick_labels = [f'{c1:1.2f}' for c1 in x_data]
        plt.gca().set_xscale('log', base=2)
        plt.gca().set_xticks(tick_positions)
        plt.gca().set_xticklabels(tick_labels)

    # Apply plot labels and title, with optional overrides from plot_kw (TOML)
    ax.set_xlabel(plot_kw.get("xlabel", d_c.capitalize()))
    ax.set_ylabel(plot_kw.get("ylabel", 'Response'))
    title_within_axes(plot_kw.get("title", 'Hill Fit with 95% Credible Interval'), **plot_kw)

    # Set y-axis to percent format if "ylabel" includes "%"
    if '%' in plot_kw.get("ylabel", ''):
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=""))

    if 'yaxis' not in plot_kw.get('figures',{'mcmc_curvesaxis':['yaxis']}).get('mcmc_curvesaxis',['yaxis']):
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['left'].set_visible(False)


def annotate_nd50_curve(nd50_mean, nd50_lower_bound, nd50_upper_bound, ax, alpha, color, fkw, nd50_limit=1/4):
    if nd50_mean > nd50_limit:
        return
    marker = fkw.get('nd50_plot_marker', 'o')
    markersize = fkw.get('nd50_plot_marker_size', 0)
    capsize = fkw.get('nd50_plot_capsize', 5)
    errorbar_y = fkw.get('nd50_y_position_on_plot', 0.05)
    try:
        ax.errorbar(nd50_mean, errorbar_y, xerr=[[max(0,nd50_mean - nd50_lower_bound)], [nd50_upper_bound - nd50_mean]],
                fmt=marker, color=color, capsize=capsize, markersize=markersize, alpha=alpha)
    except:
        t=1
    nx = [nd50_mean] * 2
    ny = [0, errorbar_y]
    ax.plot(nx, ny, color=color, linestyle='--', alpha=alpha)
    # Add light grey line between y = 0.5 and y = 0
    ax.plot([0, nd50_mean], [0.5, 0.5], color='lightgrey', alpha=0.8, linestyle='--')
    # Draw an arrow from (x, y) to (x, y-l1) with length l1
    yspan = ax.get_ylim()
    al = max([0.1, (yspan[1] - yspan[0])*0.05])
    ax.add_line(Line2D([nd50_mean] * 2, [.5, .5 - al], color=color, linestyle='--', alpha=alpha))
    ax.annotate('', xy=(nd50_mean, 0.5 - al), xytext=(nd50_mean, 0.5 - 0.08),
                arrowprops=dict(facecolor=color, edgecolor=color, shrinkA=0, shrinkB=0, width=0.5, headwidth=4,
                                headlength=6, alpha=alpha))


def linear_nd50_estimate(x, y):
    """
    Simplest estimate of ND50 by taking 50% of no-antibody control.

    Parameters
    ----------
    x
    y

    Returns
    -------
    closest_dilutions: x values left and right from the point that is closest to max(y)*0.5
    """

    if y.ndim > 2:
        raise ValueError("y must be 2d array of [sample at dilution, technical repeats], e.g. [7x3]")
    avg_neutralization_data = np.median(y, axis=1)
    # Calculate the difference between each neutralization data point and 0.5*max(y)
    lum_at_highest_dilution = avg_neutralization_data[np.argmin(x)]
    diff_from_50 = avg_neutralization_data - lum_at_highest_dilution*0.5

    # Find the index of the dilution points closest to 50% neutralization
    closest_index = np.argmin(np.abs(diff_from_50))
    other_closest = closest_index + 1 if diff_from_50[
                                             closest_index] > 0 else closest_index - 1  # index at the other side of the y=50% line
    if other_closest < len(x):
        # Extract the dilution points closest to 50% neutralization
        closest_indices = np.sort(np.array([closest_index, other_closest]))
        try:
            closest_dilutions = x[closest_indices]
        except:
            t=1
    else:  # 50% value is not crossed by measured points, approximate linearly
        closest_dilutions = np.array([x[closest_index], x[closest_index]/2])
    return closest_dilutions, avg_neutralization_data, lum_at_highest_dilution


def bootstrap_nd50_lin_2_dilutions(x_data, y_curve, avg_f = np.mean, debug=0):
    """
        Estimate ND50 using a linear interpolation-based bootstrap approach.

        This function identifies the two adjacent points in x_data where the average transduction
        crosses 50% (one above, one below), then enumerates all possible combinations (with replacement)
        of technical replicates at these two points. For each combination, it fits a straight line
        between the bracket points and solves for the x-value at y=0.5. The result is a distribution
        of ND50 values, from which one can compute mean/median and confidence intervals.

        Args:
            x_data (array-like): Dose or dilution values (must be in ascending or descending order).
            y_curve (2D array): Transduction measurements at each dose, shape (n_doses, n_replicates).
            avg_f (callable, optional): Function for averaging replicate measurements (default: np.mean).
            debug (bool or matplotlib.axes.Axes, optional): If True or an Axes object, debugging plots
                are generated to visualize how each bootstrap line is fit.

        Returns:
            bootstrap_nd50 (np.ndarray): Array of ND50 estimates from each bootstrap combination.
                One can derive the final ND50 estimate (e.g., mean) and confidence intervals
                (e.g., 2.5th and 97.5th percentiles) from this array.

        Example:
            # Suppose x_data = [1/32, 1/16, 1/8, 1/4], y_curve has shape (4, 3) with 3 technical replicates
            nd50_values = bootstrap_nd50_lin_2_dilutions(x_data, y_curve)
            nd50_mean = nd50_values.mean()
            nd50_ci = np.percentile(nd50_values, [2.5, 97.5])
            print(f"ND50 ~ {nd50_mean:.3f} [{nd50_ci[0]:.3f}, {nd50_ci[1]:.3f}]")
    """
    from itertools import combinations_with_replacement, product
    from scipy.stats import linregress

    def find_nd50(x_data, y_avg):
        for i in range(len(y_avg) - 1):
            if y_avg[i] <= 0.5 <= y_avg[i + 1] or y_avg[i] >= 0.5 >= y_avg[i + 1]:
                # Linear interpolation
                return i, x_data[i] + (0.5 - y_avg[i]) * (x_data[i + 1] - x_data[i]) / (y_avg[i + 1] - y_avg[i])
        return None, None

    y_avg = avg_f(y_curve, axis=1)  # Average across technical repeats
    nd50_index, nd50_a = find_nd50(x_data, y_avg)
    if nd50_index is None:
        if debug:
            return np.array([]), np.array([])
        return np.array([])
    # Generate exhaustive combinations and compute bootstrapped ND50
    repeat_indices = list(range(y_curve.shape[1]))  # [0, 1, 2]
    all_combinations = list(combinations_with_replacement(repeat_indices, len(repeat_indices)))  # Includes full sample

    bootstrap_nd50 = []
    lines_to_plot = []  # Store line information for debugging

    # Iterate over all combinations for nd50_a-1 and nd50_a+1 independently
    for combo1, combo2 in product(all_combinations, all_combinations):
        # Sample y values at nd50_a-1 and nd50_a+1
        y1_samples = y_curve[nd50_index, combo1]
        y2_samples = y_curve[nd50_index + 1, combo2]

        # Calculate means
        y1_avg = avg_f(y1_samples)
        y2_avg = avg_f(y2_samples)

        # Fit a line between the two points
        slope, intercept, _, _, _ = linregress(
            [x_data[nd50_index], x_data[nd50_index + 1]],
            [y1_avg, y2_avg]
        )

        # Evaluate x where y = 0.5 on the fitted line
        x_nd50 = (0.5 - intercept) / slope
        bootstrap_nd50.append(x_nd50)
        lines_to_plot.append((slope, intercept))

    bootstrap_nd50 = np.array(bootstrap_nd50)

    if debug:
        ax = debug if isinstance(debug, plt.Axes) else plt.subplots()[1]
        ax.scatter(np.repeat(x_data, y_curve.shape[1]), y_curve.flatten(), s=1)
        ax.plot(x_data, avg_f(y_curve, axis=1), 'r')
        ax.plot(avg_f(bootstrap_nd50), 0.5, 'ro', markersize=1)
        ax.plot([0,avg_f(bootstrap_nd50)], [0.5,0.5], '--', linewidth=1)
        # Plot all interpolated lines
        for slope, intercept in lines_to_plot[:100]:  # Limit to 100 lines for clarity
            x_range = np.linspace(x_data[nd50_index], x_data[nd50_index + 1], 100)
            y_line = slope * x_range + intercept
            ax.plot(x_range, y_line, color='blue', alpha=0.1, lw=0.5)
        if debug != ax:
            plt.show()
        return bootstrap_nd50, nd50_index

    return bootstrap_nd50


 # slope, ec50, max, min
def approximate_fraction(decimal, max_denominator=1000):
    from fractions import Fraction
    fraction = Fraction(decimal).limit_denominator(max_denominator)
    return fraction


def process_and_annotate(sample1, sample2, nd50_samples, ax, d_c, data_top=(), pairs=0, toffset=0.02, fp=None,
                         text_gap=0.05, log_base=2, nd50_thr_log=0.5, gap_in_display_coords = 0.2):
    """Clean samples, compare intervals, and annotate the plot if significant.
        gap_in_display_coords = 0.2  # Vertical gap as a proportion of the figure height

    """
    from coretia.nl_model import bayesian_threshold_test_cached as bayesian_threshold_test


    # Retrieve and clean data for the two samples
    data1 = nd50_samples[sample1]
    data2 = nd50_samples[sample2]
    if data1 is None or data2 is None:
        return 0
    clean_data1 = data1[~np.isnan(data1)]
    clean_data2 = data2[~np.isnan(data2)]
    if len(clean_data1) == 0 or len(clean_data2) == 0:
        # No ND50 in one of the curves, no need to draw significance lines
        return 0
    if d_c == 'dilution':
        clean_data1 = 1/clean_data1
        clean_data2 = 1/clean_data2
    if data_top is None or pairs:
        data_top_start = max(clean_data1.max(), clean_data2.max())
    else:
        data_top_start = data_top[-1]
    # Check if both samples have sufficient data
    if len(clean_data1) > 2 and len(clean_data2) > 2:
        # Get positions and determine midpoint
        pos1 = list(nd50_samples.keys()).index(sample1)
        pos2 = list(nd50_samples.keys()).index(sample2)
        x_pos = (pos1 + pos2) / 2

        # Get current axis limits to determine proper spacing
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        # Length of the gap for the star. Reciprocal for ADK dilution series to match MCMC distribution (e.g. sigma = 1)
        td1 = 1 / clean_data1 if d_c == 'conc_ng' else clean_data1
        td2 = 1 / clean_data2 if d_c == 'conc_ng' else clean_data2

        if nd50_thr_log == -1:
            mean_diff = np.abs(np.log2(np.mean(td1)) - np.log2(np.mean(td2)))
            ann = f'{mean_diff:.2f}' # Format difference to 2 decimal places
            fontsize = 8
        else:
            is_significant = bayesian_threshold_test(td1, td2, log_base=log_base, threshold=nd50_thr_log)
            ann = '*' if is_significant else 'ns'
            fontsize = 12 if is_significant else 8

        gap_size = text_gap * len(ann)  # Adjust as needed for appearance

        if d_c == 'dilution':
            # log scale will be set in caller, use a fixed log2 increment:
            above_bar = data_top_start * (2 ** gap_in_display_coords - 1)
            y_annotation =  data_top_start + above_bar if fp is None else fp
            offset = -toffset if ann == '*' else toffset  # Fine-tune vertical offset
            star_offset = y_annotation * (2 ** offset)
        else:
            above_bar = data_top_start * 0
            y_annotation = data_top_start + above_bar if fp is None else fp
            offset = -toffset if ann == '*' else toffset*8/12  # Fine-tune vertical offset, account for font size diff between * (12) and ns (8)
            star_offset = y_annotation + offset

        # Draw line segments before and after the star
        ax.plot([pos1, x_pos - gap_size], [y_annotation, y_annotation], color='black', linewidth=1,
                zorder=9)  # Before star
        ax.plot([x_pos + gap_size, pos2], [y_annotation, y_annotation], color='black', linewidth=1,
                zorder=9)  # After star

        # Add the star annotation
        ax.text(x_pos, star_offset, ann, ha='center', va='center_baseline', color='black', fontsize=fontsize, zorder=10)
        return y_annotation
    else:
        print(f"Not enough data samples: {len(clean_data1)}, {len(clean_data2)}")
        return 0


def sort_df_samples(df, category_pair):
    unique_samples = df.index.get_level_values('plate').unique()
    if category_pair is not None:
        # Sort based on the prefix and the specified category order,
        # i.e. if category_pair == ['Variable', 'Constant'] then sorted_samples should be:
        # ['Cirmi P148 Variable', 'Cirmi P148 Constant', 'Cirmi P254 Variable', 'Cirmi P254 Constant']
        pattern = re.compile(rf'^(.*?)(\b{"|".join(map(re.escape, category_pair))}\b)(.*)$')
        try:
            sorted_samples = sorted(unique_samples, key=lambda x: (
                pattern.match(x).group(1) + pattern.match(x).group(3),  # Combine the parts excluding the category
                category_pair.index(pattern.match(x).group(2))  # Sort by category
            ))
        except:
            t = 1
    else:
        sorted_samples = unique_samples
    return sorted_samples


def linear_bootstrap(df, d_c, category_pair, avg_f = np.mean, out_path = None, plot_kw={}):
    from coretia.plot import linear_bootstrap_zoomed, get_colors_for_curves
    from coretia.bayesian import linear_nd50_slope_in_log2_space_estimates

    sorted_samples = sort_df_samples(df, category_pair)
    colors, linestyles = get_colors_for_curves(len(sorted_samples), category_pair, sorted_samples, plot_kw, color_key='allcurvescolors')

    bootstrap_nd50 = dict.fromkeys(sorted_samples)
    nd50_stats = dict.fromkeys(sorted_samples)
    for i, sample in enumerate(sorted_samples):
        sample_data = df.loc[sample].groupby(d_c)['Lum']
        x = df.loc[sample][d_c].unique()
        try:
            # Find the minimum array length in the list
            original_y = [group.values for _, group in sample_data]
            min_length = min(len(arr) for arr in original_y)  # Returns 2

            # Trim arrays longer than minimum length
            trimmed_list = [arr[:-1] if len(arr) > min_length else arr for arr in original_y]
            y_curve = np.array(trimmed_list)
        except:
            print(f"Likely inhomogenous number of repeats across dilutions for {sample}:{df.loc[sample]}")
            raise
        # Remove no-antibody control from normalized curve data
        if d_c == 'dilution':
            abf_pos = np.where(x == 1)
        elif d_c == 'conc_ng':
            abf_pos = np.where(x == 0)
        else:
            abf_pos = []  # There is no positive reference, no need to remove value
        x = np.delete(x, abf_pos)
        y_curve = np.delete(y_curve, abf_pos, axis=0)
        x = np.sort(x)  # after removing pos control, sort to match y ordering

        repeats = np.array([d1.shape[0] for d1 in y_curve])
        n_samp = repeats.min()
        if any(repeats != y_curve[0].shape[0]):
            print(
                f"Not all points on the curve have same number of repeated samplings: {repeats}. Retaining only {n_samp} points.")
        y_curve = np.array([y1[:n_samp] for y1 in y_curve])
        bootstrap_nd50[sample] = 2 ** bootstrap_nd50_lin_2_dilutions(np.log2(x), np.array(y_curve), avg_f=avg_f)
        lm = linear_nd50_slope_in_log2_space_estimates(np.log2(x), avg_f(y_curve, axis=1))
        if lm[0] is None:
            x5 = low = high = 1  # placeholder if ND50 cannot be interpolated as curve never crosses 50%
        else:
            x5 = 2 ** lm[0][0]
            if len(bootstrap_nd50[sample]):
                low, high = np.percentile(bootstrap_nd50[sample], [2.5, 97.5])
            else:  # no bootstrapped lines, use mean  (CI=0)
                low = high = x5

        nd50_stats[sample] = (x5, low, high)
        if out_path is not None:
            op1 = out_path.with_name(out_path.stem + f"_{sample}_linear.png")
            linear_bootstrap_zoomed(x, np.array(y_curve), colors[i], op1, None, 6, dpi=plot_dpi)

    # Set 1 as placeholder for ND50 > 1
    x5 = [bv.mean() if bv is not None and len(bv) else 1 for bv in bootstrap_nd50.values()]
    low_high = [np.percentile(bv, [2.5, 97.5]) if bv is not None and len(bv) else [np.nan, np.nan] for bv in
                bootstrap_nd50.values()]
    nd50_stats = {k1: (x51, lh[0], lh[1]) for k1, x51, lh in zip(sorted_samples, x5, low_high)}
    return nd50_stats, bootstrap_nd50


def nd50_barplot_mcmc(subjects, nd50_data, nd50_samples, d_c, colors, hatch=None, plot_kw=None,
                      categories=None, log_base=2, nd50_thr_log=0.5):
    """
    Create a bar plot for ND50 values with p-value brackets drawn without altering the y-axis.

    Args:
    subject (list): ['TRX-032', 'TRX-033', 'TRX-035', 'TRX-036', 'TRX-037', 'TRX-040',
       'TRX-042', 'TRX-044', 'TRX-045']
    nd50_data (dict, each 3 values): ND50 median, lower, upper Credible Interval values for each plate.
    nd50_samples (dict, each numpy vector): ND50 samples for each plate
    d_c: 'dilution' or 'conc_ng'
    colors: list of color definitions for each plate
    hatch: list of strings (hatch for mpl.plot)
    plot_kw
    plates: list of keys to nd50_data and nd50_samples. Use when you want to plot a subset of bars or specify non-alphabetic ordering.
    """
    from matplotlib.ticker import FuncFormatter
    label_rotation = plot_kw.get('figures', {'label_rotation':40}).get('label_rotation', 40)
    minor_label_rotation = plot_kw.get('figures', {'minor_label_rotation':0}).get('minor_label_rotation', 0)
    label_ha = plot_kw.get('figures', {'label_ha':'right'}).get('label_ha', 'center')
    fixed_bar_y_positions =  plot_kw.get('figures', {'mcmc_barplot_significance_positions': ()}).get('mcmc_barplot_significance_positions', ())
    plates = list(nd50_data.keys())
    colors = colors if colors is not None else 'skyblue'
    error, nd50means = nd50_stats(plates, d_c, nd50_data)

    bar_relative_width = plot_kw.get('figures', {'bar_width': 0.8}).get('bar_width', 0.8)  # Bar width as proportion of available space
    bar_spacing = plot_kw.get('figures', {'bar_spacing': 1}).get('bar_spacing', 1)
    bar_pos = np.arange(len(plates)) * bar_spacing
    fig,ax = plt.subplots(figsize=plot_kw.get('figures',{}).get('barplotsize',(6,6)))

    def reciprocal_formatter(y, pos):
        return f'1/{int(y)}'

    # draw hatch
    if hatch is not None:
        ax.bar(bar_pos, nd50means, width=bar_relative_width, color=None, alpha=0.8, hatch=hatch, lw=1., zorder=0)
    bars = ax.bar(bar_pos, nd50means, width=bar_relative_width, color=colors, alpha=0.8)

    # Plot error bars with darker colors
    for i, bar in enumerate(bars):
        # Get the x position and height of the bar
        x = bar.get_x() + bar.get_width() / 2  # Center of the bar
        height = bar.get_height()  # Height of the bar
        error_bar_color = darken_color(colors[i], factor=0.7)
        try:
            ax.errorbar(x, height, yerr=np.array(error[:, i],ndmin=2).T, color=error_bar_color, fmt='none', capsize=0, elinewidth=1.5)
        except:
            t=1
    # Set plot labels and title
    ax.set_ylabel(plot_kw.get("xlabel", 'Neutralization'))  # Barplot y label uses x label of raw dataset (dilution or conc_ng)
    toffset = plot_kw.get('figures', {}).get('barplot_toffset', 0.04)  # scales with y axis limits, this is good for 1/128
    text_gap = plot_kw.get('figures', {}).get('barplot_text_gap', 0.05)
    gap_in_display_coords = plot_kw.get('figures', {}).get('barplot_vertical_gap', 0.2)
    # If pairs of curves were analyzed, compare their nd50 credible intervals and put * if significant
    if len(plates) == 2 * len(subjects):
        # Calculate midpoints for pairs of bars (two plates per subject)
        xtick_positions = np.arange(len(subjects)) * 2 + 0.5  # ticks between pairs of bars
        xticklabels = plot_kw.get('figures', {}).get('nd50_labels', subjects)
        if len(xticklabels) != len(subjects):
            raise RuntimeError('Number of subjects does not match number of nd50_labels defined in TOML file')
        xticklabels = [xl1.encode('utf-8').decode('unicode_escape') for xl1 in xticklabels]
        if plot_kw.get('figures', {}).get('mcmc_barplot_significance', 'star') == 'star':
            data_top = []
            for i1, s1 in enumerate(subjects):
                samples = [p for p in plates if s1 in p]
                if len(samples) == 2:
                    # Process and annotate for the pair of plates
                    if len(fixed_bar_y_positions):
                        if fixed_bar_y_positions == 'max':
                            fp = nd50means.max() + 1.5 * error[1, np.argmax(nd50means)]
                        elif len(fixed_bar_y_positions) == len(subjects):
                            fp = fixed_bar_y_positions[i1]
                        else:
                            raise RuntimeError(f"Number significance bracket y positions {fixed_bar_y_positions} is different from number of subjects {subjects}.")
                    else:
                        fp = None
                    data_top.append(
                        process_and_annotate(samples[0], samples[1], nd50_samples, ax, d_c, pairs=1, toffset=toffset,
                                             fp=fp, text_gap=text_gap, nd50_thr_log=nd50_thr_log, gap_in_display_coords=gap_in_display_coords))
                    if data_top[-1] is None:
                        process_and_annotate(samples[0], samples[1], nd50_samples, ax, d_c, pairs=1, toffset=toffset,
                                             fp=fp, text_gap=text_gap, nd50_thr_log=nd50_thr_log, gap_in_display_coords=gap_in_display_coords)
                else:
                    raise ValueError(f"Plate names {plates} should contain the sample name {s1}, e.g. 'TRX-032 2%' and 'TRX-032 10%'")
        # Add legend to explain the colors
        import matplotlib.patches as mpatches
        legendtexts = plot_kw.get('figures', {}).get('barplot_legend', [])
        if len(legendtexts):
            if len(legendtexts) != len(np.unique(colors)):
                raise RuntimeError(f"Custom text for legend {legendtexts} should match colors {np.unique(colors)}. Check TOML file.")
            legend_handles = [mpatches.Patch(color=c1, label=l1) for l1, c1 in zip(legendtexts, colors)]
            ax.legend(handles=legend_handles, title="Condition", loc="best")

    # Check if all combinations should be analyzed
    elif plot_kw.get('figures', {}).get('mcmc_nd50_all_combinations', 0) or \
         plot_kw.get('figures', {}).get('mcmc_nd50_pair_with_next', 0):
        # Generate all combinations of the samples
        if plot_kw.get('figures', {}).get('mcmc_nd50_all_combinations', 0):
            sample_combinations = combinations(nd50_samples.keys(), 2)
            pairs = 0
            data_top = [nd50means.max()]
        else: # When e.g. effect of time is analyzed, compare only one to the next, but not all combinations
            keys = nd50_samples.keys()
            sample_combinations = list(zip(keys, list(keys)[1:]))
            pairs = 1  # put * and line over pairs of bars
            data_top = []
        # Process and annotate for each combination
        for i1, (sample1, sample2) in enumerate(sample_combinations):
            data_top.append(
                process_and_annotate(sample1, sample2, nd50_samples, ax, d_c, data_top=data_top, pairs=pairs,
                                     toffset=toffset, text_gap=text_gap, log_base=log_base, nd50_thr_log=nd50_thr_log, gap_in_display_coords=gap_in_display_coords))

        # Optionally adjust xtick positions if needed
        xticklabels = plot_kw.get('figures', {}).get('nd50_labels', list(nd50_samples.keys()))
        xtick_positions = (np.array([i for i in range(len(xticklabels))]) ) * bar_spacing

    elif plot_kw.get('figures', {}).get('nd50_labels') is not None:
        xticklabels = plot_kw['figures']['nd50_labels']
        xtick_positions = np.array([i for i in range(len(xticklabels))]) + 0.5  # minor and major ticks are 0.5 units shifted (otherwise major overwrites minor)
        data_top = [nd50means.max()]
    else:  # If len(plates) == len(subjects)
        xtick_positions = np.arange(len(subjects)) * bar_spacing
        xticklabels = subjects
        data_top = [nd50means.max()]
    ax.set_xticks(xtick_positions)
    mats = plot_kw.get('figures', {'major_fontsize': 10}).get('major_fontsize', 10)
    majorticklabels = [xl1.encode('utf-8').decode('unicode_escape') for xl1 in xticklabels]
    ax.set_xticklabels(majorticklabels, rotation=label_rotation, ha=label_ha, fontsize=mats)

    # Optionally hide major label (e.g. only 1 subject with multiple categories)
    if not plot_kw.get('figures',{'major_labels_show':1}).get('major_labels_show', 1):
        ax.set_xticklabels([])
        ax.set_xticks([])

    if plot_kw.get('figures', {}).get('barplot_category_minor_xticks', False):
        if categories is None and plot_kw.get('figures',{'minor_labels':categories}).get('minor_labels') is None:
            raise RuntimeError(f"TOML figures.barplot_category_minor_xticks=1 but neither categories nor minor_labels are not defined.")
        ax.set_xticks(range(len(plates)), minor=True)
        pkw_cat = plot_kw.get('figures',{'minor_labels':categories}).get('minor_labels', categories)
        minorticklabels = [xl1.encode('utf-8').decode('unicode_escape') for xl1 in pkw_cat * len(subjects)]
        mts = plot_kw.get('figures',{'minor_fontsize':8}).get('minor_fontsize', 8)
        ax.set_xticklabels(minorticklabels, minor=True, fontsize=mts, rotation=minor_label_rotation)

        mavp = plot_kw.get('figures', {'barplot_major_vertical_pad': 20}).get('barplot_major_vertical_pad', 20)
        mivp = plot_kw.get('figures', {'barplot_minor_vertical_pad': 10}).get('barplot_minor_vertical_pad', 10)
        ax.tick_params(axis='x', which='major', pad=mavp)  # Move major tick labels down
        ax.tick_params(axis='x', which='minor', pad=mivp)

    # Apply the formatter to y-axis ticks, only integer powers of two, 1/2, 1/4...
    if d_c == 'dilution':
        ax.set_yscale('log', base=2)
        y_min = 0.9
        try:
            y_max = plot_kw.get('figures', {}).get('barplotylim', max([128, max(data_top)])*1.05)
        except:  # only one bar, data_top is empty
            y_max = ax.get_ylim()[1]
        ax.set_ylim(y_min, y_max)  # Tiny bar from .9-1 to indicate non-neutralizing
        #powers_of_2 = [2 ** i for i in range(int(np.log2(y_min)) + 1, int(np.log2(y_max)) + 1)]
        #ax.set_yticks(powers_of_2)
        ax.yaxis.set_major_formatter(FuncFormatter(reciprocal_formatter))
    elif d_c == 'conc_ng':
        try:
            y_max = plot_kw.get('figures', {}).get('barplotylim', min([1/0.05, max(data_top)])*1.05)
        except:  # only one bar, data_top is empty
            y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max)  # Tiny bar from .9-1 to indicate non-neutralizing

    if not plot_kw.get('figures',{'barplot_yaxis':1}).get('barplot_yaxis',1):
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['left'].set_visible(False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    title_within_axes(plot_kw.get('title', ''), **plot_kw)

    plt.tight_layout()
    return fig


def nd50_stats(condition_labels, d_c, nd50_data):
    nd50means = np.array([np.nanmean(nd50_data[k1][0]) for k1 in condition_labels])
    nd50low = np.array([np.nanmean(nd50_data[k1][1]) for k1 in condition_labels])
    nd50high = np.array([np.nanmean(nd50_data[k1][2]) for k1 in condition_labels])
    if d_c == 'dilution':
        nd50means = 1 / nd50means  # higher bar should show higher neutralization
        nd50high_a = nd50high
        nd50high = 1 / nd50low  # swap high and low as 1/value operation swaps relations
        nd50low = 1 / nd50high_a
    # Calculate error bars
    error_lower = nd50means - nd50low  # Difference between mean and lower bound
    error_upper = nd50high - nd50means  # Difference between high bound and mean
    error = np.array([error_lower, error_upper])  # Asymmetrical error bars
    return error, nd50means


if __name__ == '__main__':
    pass
