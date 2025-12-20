from Utils import *
from preproceed import *
from Plot import *
from Analyzer import Analyzer
from CSP import *
import matplotlib.pyplot as plt

# Load data
preprocess = dataLoader("B0102T")
data, labels = preprocess.X, preprocess.y # data shape: (n_trials, n_channels, n_times), labels shape: (n_trials,)
print(f"Data shape: {data.shape}")
data

# Run pipeline
data_analyzer = Analyzer(fs=preprocess.fs)
data_analyzer.run(data, remove_erp=True)

# define baseline (example: -1s to 0s)
baseline_start = int(0.0 * preprocess.fs)
baseline_end   = int(1.0 * preprocess.fs)

data_analyzer.apply_baseline(baseline_start, baseline_end)
data_analyzer.apply_spatial_filter(method='laplacian')

# CWT
cwt_alpha, freq_alpha, time_alpha = cwt_analysis(data_analyzer.erd_bandpassed[0,:,:], fs=preprocess.fs)
cwt_beta, freq_beta, time_beta = cwt_analysis(data_analyzer.ers_bandpassed[0,:,:], fs=preprocess.fs)

# CSP pipeline

# # Plot ERD/ERS
# fig = merge2Signals(
#     data_analyzer.alpha_percent,  
#     data_analyzer.beta_percent,   
#     fs=preprocess.fs,
#     label1="Alpha ERD (%) - Laplacian",
#     label2="Beta ERS (%) - Laplacian"
# )

# figs = plot2Signal(
#     data_analyzer.alpha_percent,
#     data_analyzer.beta_percent, 
#     fs=preprocess.fs,
#     label1="Alpha ERD (%) - Laplacian",
#     label2="Beta ERS (%) - Laplacian",
# )

# fig2 = plot_topographic_map(
#     data_analyzer.alpha_percent,
#     fs=preprocess.fs,
#     time_points=[0, 1, 2, 3],
#     title="Alpha ERD Topography (Laplacian)"
# )

# plot_combined_scalograms(cwt_alpha, freq_alpha, time_alpha, cwt_beta, freq_beta, time_beta)

mi_start = int(5.0 * preprocess.fs)  # 1s after cue at t=4s
mi_end   = int(7.0 * preprocess.fs)  # 3s after cue

features = data_analyzer.motorImagery(mi_start, mi_end)
print(f"Extracted {features.shape[1]} features from {features.shape[0]} trials")

plt.show()

