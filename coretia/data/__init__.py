import os
import platform
import subprocess
from pathlib import Path

import numpy as np
from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot as plt

from coretia import process
from coretia.plot import figure_as_word, plot_dpi, title_within_axes


def execute_scripts(base_dir, script_name=None, full_script_path=None, additional_args=None):
    """
    Discover and execute scripts or use a provided full script path with specified arguments.

    Parameters:
        base_dir (str): The base directory to start the search.
        script_name (str, optional): The name of the script to search for in directories (e.g., 'generate_fig.py').
        full_script_path (str, optional): A full path to a specific script to execute (e.g., 'process.py').
        additional_args (list, optional): Additional arguments to pass when executing the script.

    Notes:
        - If `script_name` is provided, it searches for the script in the directory tree and executes it.
        - If `full_script_path` is provided, it executes that script directly with `additional_args`.
    """
    if not script_name and not full_script_path:
        print("Error: Either 'script_name' or 'full_script_path' must be provided.")
        return

    scripts_to_execute = []
    errors = []
    max_depth = 1  # Set maximum depth (depth 0 is base_dir)

    # Collect scripts based on folder content with depth limit
    for root, dirs, files in os.walk(base_dir):
        rel_path = os.path.relpath(root, base_dir)
        depth = rel_path.count(os.sep)
        if depth >= max_depth:
            dirs[:] = []  # Prevent os.walk from recursing further
            continue

        toml_files = [f for f in files if f.endswith('.toml')]
        if toml_files:
            scripts_to_execute.append((full_script_path, ["recursive", root]))
        if script_name in files:
            scripts_to_execute.append((os.path.join(root, script_name), []))

    # Execute collected scripts
    for script_path, args in scripts_to_execute:
        print(f"Executing script: {script_path} with arguments: {args}")
        try:
            result = subprocess.run(['python', script_path] + args, check=True, text=True, capture_output=True)
            print(f"Output of {script_path}:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script_path}:")
            print(e.stderr)
            errors.append(f"Error with: {script_path}, {args}\n")
    return errors


def run(base_dir, full_script_path=process.__file__):
    """
    Run process.py with discovered folders containing TOML files or with 'recursive' argument,
    and execute generate_fig.py scripts in discovered folders.

    Parameters:
        base_dir (str): The base directory to start the search.
        full_script_path (str): The full path to 'process.py'. Defaults to the file path of 'process' module.
    """
    # Run process.py with discovered folders or 'recursive'
    errors = execute_scripts(
        base_dir=base_dir,
        full_script_path=full_script_path,
        additional_args=None,  # Logic for TOML check is within `execute_scripts`
        script_name='generate_fig.py')
    print(f"Found errors: {errors}")


def get_font(size):
    if platform.system() == "Windows":
        return ImageFont.truetype("arial.ttf", size)
    elif platform.system() == "Darwin":  # macOS
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    else:  # Linux
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def adjust_bayes_linear_barplot_name(panels, script_dir):
    """
    Check if Bayes plot exists, revert to linear-bootstrap if not.
    Supports nested lists.
    """
    def process_element(original_name):
        img_path = resolve_image_path(original_name, script_dir)
        if '_bayes' in original_name and not img_path.exists():
            new_name = original_name.replace('_bayes', '')
            new_path = resolve_image_path(new_name, script_dir)
            if new_path.exists():
                return new_name
            else:
                raise FileNotFoundError(
                    f"Neither Hill-MCMC nor linear-bootstrap version found for {original_name} at {new_path}"
                )
        else:
            if not img_path.exists():
                raise FileNotFoundError(f"File not found: {img_path}")
            return original_name

    for i in range(len(panels)):
        panels[i] = nested_apply(panels[i], process_element)


def parse_image_dimensions(panels, scale_factors, script_dir):
    """
    Calculate dimensions of the figure and panels.
    Supports nested lists for vertical stacking.
    """
    row_image_widths, panel_heights, dpi_list = [], [], []

    def get_img_size(img, row_idx, col_idx):
        img_path = resolve_image_path(img, script_dir)
        with Image.open(img_path) as img_obj:
            width, height = img_obj.width, img_obj.height
            scale = scale_factors.get((row_idx, col_idx), 1) if scale_factors else 1
            width, height = int(width * scale), int(height * scale)
            dpi_list.append(img_obj.info.get('dpi', (72,))[0])
            return width, height

    for row_idx, row in enumerate(panels):
        row_widths, row_heights = [], []

        for col_idx, img in enumerate(row):
            if isinstance(img, list):
                nested_widths = []
                nested_heights = []
                for nested_img in img:
                    w, h = get_img_size(nested_img, row_idx, col_idx)
                    nested_widths.append(w)
                    nested_heights.append(h)
                row_widths.append(max(nested_widths))
                row_heights.append(sum(nested_heights))
            else:
                w, h = get_img_size(img, row_idx, col_idx)
                row_widths.append(w)
                row_heights.append(h)

        row_image_widths.append(row_widths)
        panel_heights.append(max(row_heights) if row_heights else 0)

    dpi = max(set(dpi_list), key=dpi_list.count) if dpi_list else 72
    total_width = max(map(sum, row_image_widths)) if row_image_widths else 0
    total_height = sum(panel_heights)
    figsize = (total_width / dpi, total_height / dpi)
    return dpi, figsize, panel_heights, row_image_widths, total_width, total_height


def draw_panel_element(img, width, x_offset, y_offset, total_width, total_height, fig,
                       row_height, label_idx, scale_factors, row_idx, col_idx,
                       script_dir, subpanel_labels):
    """
    Draw a single panel element (or a nested list of elements) at the given position.
    """
    if isinstance(img, list):
        current_y = y_offset
        for nested_img, nested_height in zip(img, width['heights']):
            draw_panel_element(
                nested_img,
                {'width': width['width'], 'heights': [nested_height]},
                x_offset,
                current_y,
                total_width,
                total_height,
                fig,
                nested_height,
                label_idx,
                scale_factors,
                row_idx,
                col_idx,
                script_dir,
                subpanel_labels
            )
            current_y -= nested_height / total_height
    else:
        img_path = resolve_image_path(img, script_dir)
        with Image.open(img_path) as img_obj:
            scale = scale_factors.get((row_idx, col_idx), 1) if scale_factors else 1
            resized_img = img_obj.resize(
                (int(img_obj.width * scale), int(img_obj.height * scale)),
                Image.Resampling.LANCZOS
            ) if scale != 1 else img_obj
            ax = fig.add_axes([
                x_offset / total_width,
                (y_offset - row_height / total_height),
                width['width'] / total_width,
                row_height / total_height
            ])
            ax.imshow(resized_img)
            ax.axis('off')
            if subpanel_labels and 0 <= label_idx[0] < len(subpanel_labels):
                label_x = (x_offset / total_width) + (0.01 * (width['width'] / total_width))
                label_y = y_offset - (0.01 * (row_height / total_height))
                fig.text(label_x, label_y, subpanel_labels[label_idx[0]],
                         fontsize=16, fontweight='bold', ha='left', va='top', color='black')
                label_idx[0] += 1


def merge_panels_into_word(panels, script_dir, imgname, caption_docx, scale_in_word=1, scale_factors=None,
        return_png=False, subpanel_label_start='A', subpanel_label_skip=0, keep_img=False, ext='png'):

    """
    Create a composite figure from a nested panels structure and export to Word.
    Supports nested lists for vertical stacking.
    Subpanel_label_skip: jump over the first n panels without putting a letter (in case those already are lettered)
    """

    adjust_bayes_linear_barplot_name(panels, script_dir)

    # Generate subpanel labels (A, B, C, ...)
    def count_subpanels(panels):
        count = 0
        for el in panels:
            if isinstance(el, list):
                count += count_subpanels(el)
            else:
                count += 1
        return count

    subpanel_labels = []
    if subpanel_label_start:
        start_index = ord(subpanel_label_start.upper()) - 65# + subpanel_label_skip
        total_panels = count_subpanels(panels)
        #subpanel_labels = [chr(65 + start_index + i) for i in range(total_panels)]
        # Generate labels for all panels (excluding skipped ones)
        subpanel_labels = [None] * subpanel_label_skip + [
            chr(65 + start_index + i) for i in range(total_panels - subpanel_label_skip)
        ]

    dpi, figsize, panel_heights, row_image_widths, total_width, total_height = \
        parse_image_dimensions(panels, scale_factors, script_dir)

    fig = plt.figure(figsize=figsize)
    label_idx = [0]  # This will track which panel we're on
    y_offset = 1.0  # Start from the top

    for row_idx, (row, row_height, widths) in enumerate(zip(panels, panel_heights, row_image_widths)):
        x_offset = 0
        for col_idx, (img, width) in enumerate(zip(row, widths)):
            if isinstance(img, list):
                # Calculate heights for nested images
                nested_heights = []
                for nested_img in img:
                    nested_img_path = resolve_image_path(nested_img, script_dir)
                    with Image.open(nested_img_path) as nested_img_obj:
                        scale = scale_factors.get((row_idx, col_idx), 1) if scale_factors else 1
                        nested_heights.append(int(nested_img_obj.height * scale))

                draw_panel_element(
                    img,
                    {'width': width, 'heights': nested_heights},
                    x_offset,
                    y_offset,
                    total_width,
                    total_height,
                    fig,
                    sum(nested_heights),
                    label_idx,
                    scale_factors,
                    row_idx,
                    col_idx,
                    script_dir,
                    subpanel_labels
                )
            else:
                draw_panel_element(
                    img,
                    {'width': width, 'heights': [row_height]},
                    x_offset,
                    y_offset,
                    total_width,
                    total_height,
                    fig,
                    row_height,
                    label_idx,
                    scale_factors,
                    row_idx,
                    col_idx,
                    script_dir,
                    subpanel_labels
                )
            x_offset += width
        y_offset -= row_height / total_height

    # Save the image
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(imgname, bbox_inches='tight', pad_inches=0, dpi=dpi)
    if return_png:
        return imgname
    figure_as_word(imgname, caption_docx, scale=scale_in_word, keep_img=keep_img, ext=ext)


def resolve_image_path(img, script_dir):
    """Resolve image path based on whether it starts with a dot."""
    import coretia.process
    if str(img).startswith('.') or hasattr(coretia.process, 'out_base'):
        return Path(script_dir / img)
    else:
        return script_dir / 'output' / img


def merge_row_images_into_word(panels, script_dir, imgname, caption_docx=None, scale_in_word=1.0,
                               return_png=False, subpanel_label_start = 'A', ext='png', keep_img=False):
    """
    Stitch images horizontally and vertically using Matplotlib and add labels with fig.text.
    """
    # Adjust names for all rows first
    adjust_bayes_linear_barplot_name(panels, script_dir)

    # --- Calculate layout using Matplotlib ---
    all_image_data = []
    row_image_widths_px = []
    row_max_heights_px = []
    max_total_width_px = 0
    dpi = plot_dpi # Use a consistent DPI

    for row_idx, row in enumerate(panels):
        current_row_images = []
        current_row_widths = []
        current_row_heights = []
        for img_path_str in row:
            img_path = resolve_image_path(img_path_str, script_dir)
            if not img_path.exists():
                 raise FileNotFoundError(f"Image file not found: {img_path}")
            try:
                # Use matplotlib's imread to load image data for imshow
                img_data = plt.imread(img_path)
                # Get dimensions from loaded data shape (height, width, channels)
                h, w = img_data.shape[:2]
                current_row_images.append(img_data)
                current_row_widths.append(w)
                current_row_heights.append(h)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Optionally skip or add placeholder
                continue

        if not current_row_widths: # Skip empty rows
            continue

        all_image_data.append(current_row_images)
        row_image_widths_px.append(current_row_widths)
        row_max_heights_px.append(max(current_row_heights) if current_row_heights else 0)
        max_total_width_px = max(max_total_width_px, sum(current_row_widths))

    total_height_px = sum(row_max_heights_px)

    if total_height_px == 0 or max_total_width_px == 0:
        print("Warning: No images found or loaded, cannot create figure.")
        return None if return_png else None

    # Calculate figure size in inches
    figsize = (max_total_width_px / dpi, total_height_px / dpi)
    fig = plt.figure(figsize=figsize)

    # --- Place images and labels ---
    current_y_px = total_height_px # Start placing from the top
    label_idx = 0
    start_ord = ord(subpanel_label_start.upper()) - 65 if subpanel_label_start else -1

    for row_idx, row_images in enumerate(all_image_data):
        row_widths = row_image_widths_px[row_idx]
        row_max_h = row_max_heights_px[row_idx]
        current_y_px -= row_max_h # Move down by the height of the current row
        current_x_px = 0

        for i, img_data in enumerate(row_images):
            img_width_px = row_widths[i]
            img_height_px = img_data.shape[0] # Get actual height

            # Calculate axes position [left, bottom, width, height] in figure coordinates (0-1)
            # Align images horizontally to the left, vertically to the top of their row space
            ax_left = current_x_px / max_total_width_px
            # Position bottom edge based on current_y_px and align top edges within the row space
            ax_bottom = (current_y_px + (row_max_h - img_height_px)) / total_height_px
            ax_width = img_width_px / max_total_width_px
            ax_height = img_height_px / total_height_px

            ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
            ax.imshow(img_data)
            ax.axis('off') # Hide axes ticks and labels

            # Add label using fig.text (coordinates relative to figure)
            if subpanel_label_start is not None:
                label = chr(65 + start_ord + label_idx)
                # Position label slightly inside the top-left corner of the axes
                # Use small offsets in figure coordinates for consistency
                label_offset_x_fig = 10 / max_total_width_px # 10 pixels offset
                label_offset_y_fig = 10 / total_height_px # 10 pixels offset

                label_x_fig = ax_left + label_offset_x_fig
                label_y_fig = ax_bottom + ax_height - label_offset_y_fig # Offset from top edge

                fig.text(label_x_fig, label_y_fig, label,
                         fontsize=16, fontweight='bold', ha='left', va='top', color='black',
                         fontfamily='Arial') # Match parameters
                label_idx += 1

            current_x_px += img_width_px

    # --- Save figure ---
    output_path = imgname # Use the final imgname directly

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the figure to free memory

    if return_png:
        return output_path
    else:
        # Extract the scale value if it's a list (use the first value for the whole figure)
        final_scale = scale_in_word[0] if isinstance(scale_in_word, list) else scale_in_word
        figure_as_word(output_path, caption_docx, scale=final_scale, ext=ext, keep_img=keep_img)
        # os.remove(output_path) # Optional: remove the png after embedding in Word


def plot_serum_concentration(file_path, output_dir, experiment = "Variable Serum", dpi=plot_dpi, show_y_axis=True, show_legend=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.ticker import PercentFormatter

    data = pd.ExcelFile(file_path)

    variable_serum_df = data.parse(experiment)

    # Process the 'Dilution' column for x-axis labels
    def format_dilution(dilution):
        if dilution == "Antibody-free":
            return dilution
        else:
            return f"1/{dilution}"

    variable_serum_df["Dilution Label"] = variable_serum_df["Dilution"].apply(format_dilution)

    # Plot the FBS in cell culture as the base
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.bar(
        variable_serum_df.index,
        variable_serum_df["FBS"],
        bottom=variable_serum_df["Blood Serum"],
        label="FBS",
        color="#FCDE9C",
    )

    # Plot the Total serum stacked on top of FBS in cell culture
    ax.bar(
        variable_serum_df.index,
        variable_serum_df["Blood Serum"],
        label="Blood Serum",
        color="#8F003B",
    )

    # Add x-ticks with proper labels
    plt.xticks(
        variable_serum_df.index,
        variable_serum_df["Dilution Label"],
        rotation=45,
        ha="right"
    )

    # Add labels, title, and legend
    plt.xlabel("Dilution", labelpad=-15)
    plt.ylabel("Total Serum Content (%)")
    plt.ylim([0,0.1])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, symbol=''))
    title_within_axes(experiment.replace(' Serum',''), plt.gca(), title_y=0.9)  # remove " Serum" from "Constant Serum" to match figure caption text "Constant" and "Variable"

    if show_legend:
        legend = plt.legend(frameon=True)
        legend.get_frame().set_facecolor('white')  # Set background color to white
        legend.get_frame().set_alpha(0.7)  # Set transparency (0.0 to 1.0)
        legend.get_frame().set_edgecolor('none')  # Remove the outline

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
   # if not show_y_axis:
   #     plt.gca().get_yaxis().set_visible(False)
   #     plt.gca().spines['left'].set_visible(False)
    plt.tight_layout()

    # Save and show the plot
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"serum_proportion_{experiment}.png", dpi=dpi)


def crop_to_match_height(images):
    """Crop images vertically to match the smallest height in the group."""
    min_height = min(img.shape[0] for img in images)
    cropped_images = []
    for img in images:
        if img.shape[0] > min_height:
            top_crop = (img.shape[0] - min_height) // 2
            cropped_images.append(img[top_crop:top_crop + min_height, :])
        else:
            cropped_images.append(img)
    return cropped_images


def crop_to_match_width(images):
    """Crop images horizontally to match the smallest width in the group."""
    min_width = min(img.shape[1] for img in images)
    cropped_images = []
    for img in images:
        if img.shape[1] > min_width:
            left_crop = (img.shape[1] - min_width) // 2
            cropped_images.append(img[:, left_crop:left_crop + min_width])
        else:
            cropped_images.append(img)
    return cropped_images


def script_out_paths(script_path, ext='png'):
    import coretia.process
    script_dir = script_path.resolve().parent
    from coretia import remap_output_path
    if hasattr(coretia.process, 'out_base'):
        out_path = remap_output_path(script_dir / 'output', coretia.process.out_base)
        imgname = out_path.parent / f'{script_dir.name}.{ext}'
    else:
        out_path = script_dir / 'output'
        imgname = out_path.parent / 'output' / f'{script_dir.name}.{ext}'

    out_path.mkdir(parents=True, exist_ok=True)

    return script_dir, out_path, imgname


def nested_apply(obj, func):
    """
    Recursively apply a function to every non-list element in a nested list structure.
    """
    if isinstance(obj, list):
        return [nested_apply(x, func) for x in obj]
    else:
        return func(obj)


def cache_data(filepath, data=None):
    """
    Save data to pickle file if provided, otherwise load from file.

    Args:
        filepath: Path to the pickle file
        data: Data to save (if None, will load from file)

    Returns:
        Either the loaded data or the saved data
    """
    import pickle
    from pathlib import Path

    filepath = Path(filepath)

    # Save mode
    if data is not None:
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return data
        except Exception as e:
            print(f"Error saving to cache: {e}")
            return data

    # Load mode
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None
    else:
        print(f"Cache file not found: {filepath}")
        return None


def add_noise(data, log_cv):
    """
        Apply log-normal multiplicative noise to simulate biological variability in cell-based assays.

        This function models noise as a log-normal distribution, where the variability scales
        proportionally with signal intensity. This is particularly suitable for luminescence-based
        assays, where biological and technical fluctuations tend to increase with higher signal levels.

        Parameters:
        -----------
        data : np.ndarray
            The input data array representing signal intensities.
        log_cv : float
            The coefficient of variation (CV) for the log-normal noise, determining the level of
            variability in the assay.

        Returns:
        --------
        np.ndarray
            A new data array with log-normal noise applied.

        Notes:
        ------
        - The noise is drawn from a log-normal distribution with a mean of 1 and standard deviation
          controlled by `log_cv`.
        - This method captures biologically realistic variability without introducing artificial
          additive noise.
    """
    # Log-normal multiplicative noise
    log_normal_noise = np.random.lognormal(mean=0, sigma=log_cv, size=data.shape)

    # Apply noise: The data is scaled by the log-normal noise
    noisy_data = data * log_normal_noise

    return noisy_data
