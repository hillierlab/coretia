from pathlib import Path
from coretia.data import merge_row_images_into_word, script_out_paths
from coretia.data.Fig5_extrapol.extrapolate_nd50 import nd50_mean_CI_converge
script_dir, out_path, pngname = script_out_paths(Path(__file__))

nd50_mean_CI_converge(out_path=out_path, true_ND50=2**5, n_comparisons = 30)

cp = script_dir/'caption.docx'

panels = [
	['extrapolate_nd50_nd5032_extrapol30.png', 'mouse_bayes_loghillplots.png','mouse_bayes_barplot.png'],
]
merge_row_images_into_word(panels, out_path, pngname, cp, scale_in_word=[0.9])
