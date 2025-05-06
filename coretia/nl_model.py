import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools

from joblib import Memory
from coretia import datapath
cachepath = datapath/'mcmc_cache'
if not cachepath.exists():
    cachepath.mkdir()
memory = Memory(cachepath, compress=1, verbose=0)

from coretia.bootstrap import linear_nd50_estimate
from coretia.bayesian import hill_fit_mcmc, extrapolate_mcmc, bayesian_threshold_test, hill_curve_log, hill_curve


@memory.cache
def hill_fit_cached(*args, **kwargs):
    return hill_fit_mcmc(*args, **kwargs)


@memory.cache
def bayesian_threshold_test_cached(*args, **kwargs):
    result = bayesian_threshold_test(*args, **kwargs)
    return result

# python
def is_hashable(o):
    try:
        hash(o)
    except TypeError:
        return False
    return True

def check_hashables(*args, **kwargs):
    for i, arg in enumerate(args):
        if not is_hashable(arg):
            print(f"Positional argument at index {i} (type: {type(arg)}) is not hashable.")
    for key, value in kwargs.items():
        if not is_hashable(value):
            print(f"Keyword argument '{key}' (type: {type(value)}) is not hashable.")

def make_hashable(array):
    # Convert input to numpy array if not already
    array = np.asarray(array)
    return (array.shape, array.dtype, array.tobytes())


class Curve(object):
    def __init__(self, x, y, name, d_c='dilution'):
        self.x = x
        self.y = y
        self.name = name
        assert isinstance(x, np.ndarray), f"x must be an ndarray but got {type(x)}"
        assert y.ndim == 2, f"y must be a [n_dilutions x n_repeats] but got {y.shape}"
        assert len(x) == len(y), f"x and y must be the same length (n_dilutions)"
        self.d_c = d_c
        if d_c == 'dilution':
            try:
                assert all(np.equal(x, x.astype(int))), f"Not all dilution factors are integers. e.g. [4,8,16..] expected."
            except:
                t = 1
                raise
            assert np.all(x > 0), f"x must contain dilution factors, e.g. [4, 8, 16...]"
            self.x1 = 1/x
        else:
            self.x1 = x

        if np.isnan(y).any():
            raise ValueError(f"NaN values detected in plate")

    def check_quality(self, threshold = 0.15):
        if self.y[0].mean() < 1-threshold:
            raise ValueError(f"{self.name}: Lowest cc (highest dilution) {self.x1[0]} (1/{self.x1[0]} is too high, curve starts below {1-threshold}")
        elif self.y[-1].mean() > threshold:
            raise ValueError(f"{self.name}: Highest cc (lowest dilution) {self.x1[-1]} (1/{self.x1[-1]} is too low, curve ends above {threshold}")

    def hill_fit(self, hill_curve_fct = hill_curve):
        closest_dilutions, y_median, nd50_y = linear_nd50_estimate(self.x1, self.y)
        bottom = np.nanmax(y_median)
        top = max(0, np.nanmin(y_median))
        nd50_guess = closest_dilutions.mean()
        slope_guess = 1  # A common starting point for slope

        # Initial curve fitting using non-linear least squares for both curves
        self.linear_guess = [slope_guess, nd50_guess, top, bottom]
        try:
            self.hill_par = curve_fit(hill_curve_fct, self.x1, y_median, p0=self.linear_guess)[0]
        except:
            self.hill_par = self.linear_guess

    def plot(self, ax, color=None,  plot_nd50 = ('ND50 linear interpolation',), hill_curve_fct = hill_curve):
        x2 = self.x1 # if hill_curve_fct == hill_curve else self.x1_log
        hp = self.hill_par
        nd50_linear = self.linear_guess[1]
        x3 = np.repeat(x2, self.y.shape[1])
        ax.scatter(x3, self.y, label=f'{self.name}', color=color, marker='.', alpha=0.5)
        label = f"{'Log10' if hill_curve_fct == hill_curve_log else ''}: {self.name}"
        ax.plot(x2, hill_curve_fct(x2, *hp), label=label, linestyle='--', color=color)
        if plot_nd50:
            if 'ND50 linear interpolation' in plot_nd50:
                ax.axvline(nd50_linear, color=color, linestyle='--', label = 'LinND50')
            if 'ND50 Hill fit' in plot_nd50 and (hill_curve_fct == hill_curve_log and hp[1] !=0) or hp[1] < 1:  # check log10 ND50 or ND50
                ax.axvline(hp[1], color=color, linestyle=':', label = f"c_fit ND50{'_log10' if hill_curve_fct == hill_curve_log else ''}")

    def hill_fit_mcmc(self):
        self.trace = hill_fit_cached(self.x1, self.y, use_weights=False)
        #self.trace = hill_fit_cached(self.x1_log, self.y, use_weights=False)

    def nd50_estimate(self, y_target=0.5):
        """
        After MCMC model evaluation, use inverse Hill curve to solve for y=0.5
        """
        self.nd50_mean, self.nd50_lower_bound, self.nd50_upper_bound, self.dilution_at_y_0_5_samples = extrapolate_mcmc(self.trace, y_target)

    def visualize_mcmc_fit(self, ax, color = None, plot_kw=None):
        """
        Visualize the Hill curve fit with observed data and 95% credible intervals.
        Visualization happens on non-log transformed x axis.

        Parameters:
        - x_data: array of dilution values
        - y_curve: 2D array of responses (7 dilutions x 3 replicates)
        - trace: posterior samples from the Hill curve model
        - curve_name: Name for the curve in the plot title
        """
        plot_kw = plot_kw or {}
        from coretia.bootstrap import visualize_mcmc_fit
        if not hasattr(self, 'trace'):
            raise RuntimeError(f"Need to call hill_fit_mcmc before visualize_hill_fit.")
        bottom_samples = self.trace.posterior['bottom'].values.flatten()
        top_samples = [1]*len(bottom_samples)
        nd50_samples = self.trace.posterior['nd50'].values.flatten()
        slope_samples = self.trace.posterior['slope'].values.flatten()

        visualize_mcmc_fit(self.x1, self.y, slope_samples, nd50_samples, top_samples, bottom_samples, ax, color=color,
                           name=self.name, nd50_mean=self.nd50_mean, nd50_lower_bound=self.nd50_lower_bound,
                           nd50_upper_bound=self.nd50_upper_bound, d_c = self.d_c, plot_kw=plot_kw)


class Comparer(object):

    def __init__(self, curves, cmp_fct=None, **cmp_kwargs):
        self.curves = curves
        self.cmp_fct = cmp_fct
        self.cmp_kwargs = cmp_kwargs
        self.i_pairs = list(itertools.combinations(range(len(curves)), 2))

    def compare_all_bayes(self, categories=None, out_path = None, plot=False, nd50_thr_log=0.25):
        def init_plot(n, base_size=(2, 3)):
            grid_rows = int(np.ceil(np.sqrt(n)))
            grid_cols = int(np.ceil(n / grid_rows))
            width = base_size[0] * grid_cols
            height = base_size[1] * grid_rows
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(width, height), squeeze=False,
                                     gridspec_kw={'hspace': 0.2, 'wspace': 0.9})
            # Set ticks and labels for bottom left axis only
            axes[-1, 0].tick_params(axis='both', which='both', top=False, right=False)
            axes[-1, 0].set_xlabel('X Label')
            axes[-1, 0].set_ylabel('Y Label')
            axes = axes.flatten()
            # Hide any unused subplots
            for i in range(n):
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['top'].set_visible(False)
            for j in range(n, len(axes) - 1):
                axes[j].axis('off')  # Turn off axis for the extra subplots

            return fig, axes

        n_curves = len(self.curves)
        if categories is None:
            pairs = list(itertools.combinations(self.curves, 2))
        else:  # only compare curves whose name is present in both categories, e.g. "Adam 2%" with "Adam 10%"
            sample_plate_dict = extract_pairs([c1.name for c1 in self.curves], categories)
            pairs = []
            for s1, v1 in sample_plate_dict.items():
                i_pair = [c1 for c1 in self.curves if c1.name in v1]
                if len(i_pair) == 1:
                    t=1
                pairs.append(i_pair)

        if plot:
            fig, axes = init_plot(len(list(pairs)))
            axes = axes.flatten()

        # Fit each curve individually
        self.nd50_bt = dict()
        self.nd50_samples = dict()
        for i, curve in enumerate(self.curves):
            print(f"Fitting curve {i + 1}/{len(self.curves)}...")
            curve.hill_fit_mcmc()
            curve.nd50_estimate()
            self.nd50_bt[curve.name] = [curve.nd50_mean, curve.nd50_lower_bound, curve.nd50_upper_bound]
            self.nd50_samples[curve.name] = curve.dilution_at_y_0_5_samples

        # Pairwise ND50 comparison
        self.nd50_pairs_pval_logs = dict()
        self.nd50_pairs_pval_texts = dict()
        for i, cp in enumerate(pairs):
            if plot:
                [c1.visualize_mcmc_fit(axes[i]) for c1 in cp]
            try:
                significant = bayesian_threshold_test_cached(cp[0].trace.posterior['nd50'].values.flatten(),
                                                      cp[1].trace.posterior['nd50'].values.flatten(),
                                                      log_base=2, threshold=nd50_thr_log)
            except:
                t=1
                return
            p_text = f"ND50 values are {'significantly different' if significant else 'not significantly different'} between {cp[0].name} and {cp[1].name}."
            print(p_text)

            # Store result
            self.nd50_pairs_pval_logs[(cp[0].name,cp[1].name)] = significant
            self.nd50_pairs_pval_texts[(cp[0].name,cp[1].name)] = f"{'*' if significant else ''}"
        if out_path is not None and plot:
            fig.savefig(out_path)
            plt.close()
        elif plot:
            plt.show(block=True)

    def compare_all(self, axes, colors=None):
        self.nd50_bt = dict()
        self.nd50_pairs_pval_logs = dict()
        self.nd50_pairs_pval_texts = dict()
        for i1, ip1 in enumerate(self.i_pairs):
            cc = [self.curves[ip11] for ip11 in ip1]
            color_pair = [colors[ip11] for ip11 in ip1] if colors is not None else None
            result = self.compare_pair(cc, visual_debug=axes[i1], colors=color_pair)
            if result[-1] is not None:
                [c1.plot(axes[i1 + 1], color=colors[ci], log_transformed=True, override_hill_par = result[3][ci]) for ci, c1 in enumerate(cc)]

            if result[3] is not None:  # proper curve comparison did run:
                self.nd50_bt[cc[0].name] = result[0][0]
                self.nd50_bt[cc[1].name] = result[0][1]
            self.nd50_pairs_pval_logs[ip1] = result[1]
            self.nd50_pairs_pval_texts[ip1] = result[2]
        # Fill in missing values (where no neutralization is detected)
        for k1 in range(len(self.curves)):
            if self.curves[k1].name not in self.nd50_bt:
                self.nd50_bt[self.curves[k1].name] = 0  # should not appear as 10**0=1 on the barplot

    def compare_pair(self, cc, visual_debug=False, colors=None, non_neutralizing_threshold = 0.7):
        if colors is None:
            colors = ['orange', 'cyan']
        nd50 = [0 if all(c1.y.flatten() > non_neutralizing_threshold) else c1.hill_par[1] for c1 in cc]

        if visual_debug:
            ### Plot raw data and pre-fits
            ax = visual_debug if isinstance(visual_debug, plt.Axes) else plt.gca()
            #add_fraction_ticks(ax, axis='x', log=True)
            for i1, c1 in enumerate(cc):
                c1.plot(ax, color=colors[i1])

        ### Evaluation of nd50 values:
        if all(cc[0].y.flatten() > 0.5) or all(cc[1].y.flatten() > 0.5):
            print('No measured value is under 50% => no neutralization.')
            results = [nd50, 0, "All points > 50%", None]

        elif cc[0].linear_guess[1] > 1/2 or cc[1].linear_guess[1] > 1/2:
            print(f'Initial ND50 guess: {cc[0].linear_guess[1], cc[1].linear_guess[1]} shows lack of measurable neutralization.')
            results = [nd50, 0, "ND50 > 1/2", None]
        else:
            kw2={}
            kw2['visual_debug'] = visual_debug
            results = self.cmp_fct(cc, **kw2)
            [ax.axvline(results[0][i2], color=colors[i2], linestyle='-.') for i2 in range(len(cc))]

        if len(results) != 4:
            raise ValueError(f"Expected [[nd50_1, nd50_2], p_value, text, [hill_par_log1, hill_par_log2], got {results}")
        return results


def extract_pairs(input_list, categories):
    # Create a defaultdict to store grouped strings with combined prefix-postfix keys
    from collections import defaultdict
    import re
    grouped_dict = defaultdict(list)

    # Iterate over each string in the input list
    for item in input_list:
        # Search for a matching category within the string
        sample_id = re.sub(r'\s+', ' ', re.sub('|'.join(categories), '', item))  # remove double space and any of the 'categories' strings
        if sample_id == item:
            print(f"Skipping plate {item}, as it does not contain a category string, {categories}")
            continue
            #category = [c1 for c1 in categories if c1 in item]
        grouped_dict[sample_id].append(item)  # Add the original string to the grouped list

    # Convert defaultdict to regular dict (optional)
    result_dict = dict(grouped_dict)
    return result_dict


if __name__ == "__main__":
    pass
