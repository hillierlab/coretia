from pathlib import Path
from coretia.data import merge_panels_into_word, merge_row_images_into_word, script_out_paths

script_dir, out_path, pngname = script_out_paths(Path(__file__))

caption_docx = script_dir/"caption.docx"

# Merge first row into one png
panels = [[str(Path(script_dir/f'constructs.png').resolve()), f'moi_nluc_flucallcurves_rawlum.png', f'moi_nlucallcurves_rawlum.png']]
scale_factors = {(0,0): 1.5}
png_r0 = merge_panels_into_word(panels, out_path, out_path/f'row1.png', None, scale_factors=scale_factors, return_png=True)

# Merge loghillplots and barplots into two pngs
panels = [[f'moi_aav1_human1_bayes_loghillplots.png', f'moi_aav5_human2_bayes_loghillplots.png', f'moi_aav9_human3_bayes_loghillplots.png',]]
png_r1_c0 = merge_row_images_into_word(panels, out_path, out_path/f'row2c0.png', None,
                              subpanel_label_start=None, return_png=True)

panels = [[f'moi_aav1_human1_bayes_barplot.png', f'moi_aav5_human2_bayes_barplot.png', f'moi_aav9_human3_bayes_barplot.png'],
]
png_r1_c1 = merge_row_images_into_word(panels, out_path, out_path/f'row2c1.png', None,
                              subpanel_label_start=None, return_png=True)

panels = [[str(png_r0)], [str(png_r1_c0)], [str(png_r1_c1)]]
# Skip first panel (which is a row with 3 panels already merged and lettered above)
merge_panels_into_word(panels, out_path, pngname, caption_docx, subpanel_label_skip=1, subpanel_label_start='D', scale_in_word=0.9)