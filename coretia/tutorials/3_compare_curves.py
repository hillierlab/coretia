import matplotlib.pyplot as plt
from coretia.datahandling import import_multiplate, normalize
from coretia.process import estimate_log_base, plot_mcmc_fits, feasible_pairs
from coretia.plot import get_colors_for_curves
from coretia.nl_model import Curve, Comparer

import numpy as np
from pathlib import Path
toml_path = Path('../data/Fig3_moi/moi_aav9_human3.toml')

x1, y1, data, df, extras, plates, d_c = import_multiplate(toml_path)

# Create Curve objects from x1, y1, and plates
curves = [Curve(x1[i1], y1[i1], name=plates[i1], d_c=d_c) for i1 in range(len(y1))]

# Initialize comparer to compare curves
comparer = Comparer(curves)

# Perform Hill fitting for all curves
[c1.hill_fit() for c1 in curves]

# Perform Bayesian comparison and plotting of all curves
plot_kw = {'figures':  {'hill_error_style': 'hill_CI95_bars'}}
# Use None for categories to get different color for each curve
colors, linestyles = get_colors_for_curves(len(y1), None, plates, plot_kw)
comparer.compare_all_bayes(nd50_thr_log=plot_kw.get('nd50_thr_log', 0.3))
plot_mcmc_fits(x1, d_c, comparer, curves, colors, plot_kw)


