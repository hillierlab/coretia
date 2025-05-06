from pathlib import Path
from coretia.data import merge_row_images_into_word, script_out_paths
from coretia.data.Fig2_MCMC import basic_comparing_neutralization_levels
from coretia import assay_quality
script_dir, out_path, pngname = script_out_paths(Path(__file__))

assay_quality.ci_cv_figure(script_dir/'nd50_ci_cv_method_paper')  # shelve does not need the extension, adds '.db' itself
basic_comparing_neutralization_levels.adk9_demo(toml_path=script_dir / 'adk9.toml', out_path=out_path, nd50_thr_log=0)
basic_comparing_neutralization_levels.adk9_demo(toml_path=script_dir / 'adk9.toml', out_path=out_path, nd50_thr_log=0.3)
basic_comparing_neutralization_levels.compare_same_nd50_different_cvs(out_path, [0.1], dilution_step=-1, seed=41, nd50_thr_log=0)
basic_comparing_neutralization_levels.compare_same_nd50_different_cvs(out_path, [0.1], dilution_step=-1, seed=41, nd50_thr_log=0.3)
if 1:  # long calculation
    basic_comparing_neutralization_levels.nd50_mean_CI_converge(out_path=out_path, cv=0.1, n_repeats=3, dilution_step=-1, seed=range(50))


cp = script_dir / 'caption.docx'

panels = [
    [f'basic_comparing_neutralization_levels_hill_fit_[0.1]_-1_None.png',  # simulated single curve
    'basic_comparing_neutralization_levels_nd50_lin_vs_hill_cv0.1_-1_nrep3_nseed50_shr0.png',  # 50 simulated curves
     f'adk9nd50thr0.png'],  # ADK curve
    [f'basic_comparing_neutralization_levels_barplot_bootstrap_[0.1]_-1_None_thr0.png',  # barplot for 1 simulated, thr=0
    'adk9nd50thr0_linear.png',  # barplot for ADK curve
     str('boostrap-hill_histogram.png'),   # histogram of human+macaque samples
     'basic_comparing_neutralization_levels_barplot_bootstrap_[0.1]_-1_None_thr0.3.png',
     'adk9nd50thr0.3_linear.png',  # barplot for ADK curve
      ]
]

merge_row_images_into_word(panels, out_path, pngname, cp, scale_in_word=[0.9, 0.9], ext='jpg', keep_png=True)


