import matplotlib.pyplot as plt
from coretia.datahandling import import_plate, normalize
from coretia.plot import create_custom_legend
from coretia.data.Fig2_MCMC.basic_comparing_neutralization_levels import nd50_3_methods, colors_nd50_methods
import numpy as np
from pathlib import Path
toml_path = Path('../data/Fig2_MCMC/adk9.toml')

from coretia.plot import plot_curves
df, extras, plates, d_c, category_pair = import_plate(toml_path)
df1, all_single_dilution = normalize(df, d_c)
plot_kw = {'ylabel': 'Transduction Efficiency (%)',
		   'xlabel': 'ADK9 concentration (ng/mL)'}

# Filter for the specific sample
filtered_df = df1[df1['sample'] == 'ADK9']

# Omit conc_ng = 0 (no-antibody control)
filtered_df = filtered_df[filtered_df['conc_ng'] != 0]

# Extract x_data (concentrations) and y_curve (technical repeats)
x_data = filtered_df['conc_ng'].unique()  # Get unique conc_ng values
# Get the Lum values for each concentration in original order (3 technical repeats)
y_curve = np.array([filtered_df[filtered_df['conc_ng'] == conc]['Lum'].head(3).values for conc in x_data])

# Linear fit and bootstrap CI
fkw0 = {'nd50_y_position_on_plot': 0.05,  # Put the ND50 uncertainty bar close to the x axis
		'mcmc_show_nd50': 0,
		'nd50_plot_marker_size': 0,  # No need to clutter with a dot, the vertical line shows ND50 anyway
		'hill_error_style': ['hill_CI95_bars'],  #
}

nd50_stats_dict= {'plot_kw': {
	'title': '',
	'xlabel': 'Antibody Concentration [ng/mL]',
	'ylabel': 'Transduction Efficiency (%)'
}}
nd50 = nd50_3_methods(None, 'green', fkw0, x_data, y_curve, d_c='conc_ng', nd50_stats_dict=nd50_stats_dict)
legends = []
titles = []
bbox = [(0.715, 0.87), (0.733, 0.76)]
ccolor = ['green'] * 2
legends.append(zip(["Raw Samples", "Hill-fit"],
				   ccolor,
				   [':', '-']
				   ))
legends.append(zip(colors_nd50_methods.keys(), colors_nd50_methods.values(), ['--'] * 3))
titles.extend(['', 'ND50 estimation:'])
create_custom_legend(legends, titles=titles, bbox=bbox, alpha=0.5)
plt.tight_layout()
plt.show()
