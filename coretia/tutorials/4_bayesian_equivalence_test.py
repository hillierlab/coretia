import matplotlib.pyplot as plt
from coretia.datahandling import import_multiplate, normalize
from coretia.process import estimate_log_base, plot_mcmc_fits, feasible_pairs
from coretia.plot import get_colors_for_curves
from coretia.nl_model import Curve, Comparer
from coretia.nl_model import bayesian_threshold_test_cached as bayesian_threshold_test

import numpy as np
from pathlib import Path
toml_path = Path('../data/Fig3_moi/moi_aav9_human3.toml')
x1, y1, data, df, extras, plates, d_c = import_multiplate(toml_path)

# Create Curve objects from x1, y1, and plates
curves = [Curve(x1[i1], y1[i1], name=plates[i1], d_c=d_c) for i1 in range(len(y1))]

# Initialize comparer to compare curves
comparer = Comparer(curves)
comparer.compare_all_bayes(nd50_thr_log=0.3)
is_significant = bayesian_threshold_test(comparer.nd50_samples['MOI10'], comparer.nd50_samples['MOI100'], log_base=2, threshold=0.3)
print(f"ND50 for MOI10 = 1/{1/comparer.nd50_bt['MOI10'][0]:1.1f}, ND50 for MOI100 = 1/{1/comparer.nd50_bt['MOI100'][0]:1.1f}")
print(f"Difference in ND50 between MOI10 and MOI100 is {'significantly larger' if is_significant else 'not larger'} than 0.3 log2 units.")

