import matplotlib
import platform
from pathlib import Path
import os
import matplotlib.pyplot as plt

line_styles = ['-', '--']
tab20 = plt.cm.get_cmap('tab20')
# Sample every second color, avoiding using nearby, similar colors
colors0 = [tab20(i) for i in range(20)][::2] + [tab20(i) for i in range(20)][1::2]
colors = colors0 * 2  # should be enough for 40 samples

datapath = Path(os.path.dirname(__file__))/'data'
if not datapath.exists():
    raise FileNotFoundError('Data folder is missing, should be under neutralization/data')

def outpath_from_toml(toml_path:Path, suffix='', extension='.pdf', remap_to = None):
    out_path = toml_path.parent / 'output'
    if remap_to is not None:
        if remap_to.exists():
            out_path = remap_output_path(out_path, remap_to)
        else:
            print(f"Found alternative output path but it doesn't exist: {remap_to}")
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path / (toml_path.stem + suffix + extension)

    return out_path

def remap_output_path(original_path:Path, new_base_path:Path):
    if new_base_path is None:
        return original_path
    # Find the deepest folder in the new_base_path
    deepest_folder_in_new_base = new_base_path.name

    # Locate the deepest folder in the original path
    # and split the path into the part before and after it
    original_parts = list(original_path.parts)
    try:
        # Find the index of the deepest folder in the original path
        split_index = original_parts.index(deepest_folder_in_new_base)

        # Replace the part before the deepest folder with the new_base_path
        new_path = new_base_path / Path(*original_parts[split_index + 1:])

        # Get the directory part of the new path
        new_directory = new_path.parent
    except ValueError:
        raise ValueError(
            f"The deepest folder '{deepest_folder_in_new_base}' from the new base path was not found in the original path.")
    new_directory.mkdir(exist_ok=True, parents=True)
    return new_directory