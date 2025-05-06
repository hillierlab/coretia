import matplotlib.pyplot as plt
from coretia.datahandling import import_plate
import wellmap
from pathlib import Path
toml_path = Path('../data/Fig2_MCMC/adk9.toml')


### Load data and show layout of samples on the 96-well plate
fig = wellmap.show(Path(toml_path))
fig.show()
plt.close()

######################
### Plot raw luminescence values
from coretia.plot import plot_curves
df, extras, plates, d_c, category_pair = import_plate(toml_path)

plot_kw = {'ylabel': 'Raw Light Units (RLU)',
		   'xlabel': 'ADK9 concentration (ng/mL)'}

# Group data along samples (each sample is one TOML file, representing a wellmap plate)
df_raw = df.groupby(['plate']).apply(lambda x: x)
plot_curves(df_raw, d_c, None, plot_kw, category_pair)
plt.show()

######################
### Plot luminescence values normalized to antibody-free control
from coretia.datahandling import normalize
df1, all_single_dilution = normalize(df, d_c)
plot_kw['ylabel'] = 'Transduction Efficiency (%)'
plot_curves(df1, d_c, None, plot_kw, category_pair, include_ab_free=True)
plt.show()

