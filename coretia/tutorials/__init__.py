import numpy as np
import pandas as pd
from pathlib import Path
import os

from coretia.data import add_noise

out_path = Path(os.path.dirname(__file__))/'output'
out_path.mkdir(exist_ok=True)

neutralization_types = {
    'mild': {'slope':0.5, 'nd50':1/2, 'top':1, 'bottom':0.5},
    'strong':{'slope':1.2, 'nd50':1/16, 'top':1, 'bottom':0.1}
}

# Function to generate synthetic data for flat, mild, and strong neutralizing curves
def generate_synthetic_data(x_data, n_samples=1, n_repeats=3, cv=0.1, neutralization_type='flat'):
    from coretia.bayesian import hill_curve
    y_data = []
    n_dilutions = len(x_data)  # Number of dilutions based on x_data

    for _ in range(n_samples):
        if neutralization_type == 'flat':
            # Flat curve: a horizontal line with some noise
            y_curve = np.ones((n_dilutions, n_repeats))
        elif isinstance(neutralization_type, dict) or neutralization_type in neutralization_types:
            if isinstance(neutralization_type, dict):
                if all(key in neutralization_type for key in ['slope','nd50','top','bottom']):
                    hill_curve_values = hill_curve(x_data[:, np.newaxis], **neutralization_type)
                else:
                    raise ValueError(f"Custom Hill curve for generating synthetic data requires dict with keys: 'slope','nd50','top','bottom'. Got: {neutralization_type} ")
            else:  # 'mild' or 'strong'
                # Generate hill curve values with shape (n_dilutions, 1)
                hill_curve_values = hill_curve(x_data[:, np.newaxis], **neutralization_types[neutralization_type])
            # Repeat the values across all repeats to get shape (n_dilutions, n_repeats)
            y_curve = np.tile(hill_curve_values, (1, n_repeats))
        else:
            raise ValueError("Invalid neutralization_type. Choose from 'flat', 'mild', or 'strong'.")

        # Add noise to the curve
        y_curve = add_noise(y_curve, cv)
        y_data.append(y_curve)

    # Return shape (n_samples, n_dilutions, n_repeats)
    return np.array(y_data)
