from pathlib import Path
from coretia.data import merge_panels_into_word, script_out_paths

script_dir, out_path, pngname = script_out_paths(Path(__file__))

cp = script_dir/'caption.docx'
panels = [
[f'heat_inactivation_bayes_barplot.png', f'mix_inctime_bayes_barplot.png', f'24h_48h_bayes_barplot.png']
]
merge_panels_into_word(panels, out_path, pngname, cp, scale_in_word=0.6)
