from Utils import *
from preproceed import *
from Plot import *
from Analyzer import Analyzer
import matplotlib.pyplot as plt
import numpy as np

# Load data
preprocess = dataLoader("B0102T")
data, labels = preprocess.X, preprocess.y # data shape: (n_trials, n_channels, n_times), labels shape: (n_trials,)
print(f"Data shape: {data.shape}")

# Run pipeline
data_analyzer = Analyzer(fs=preprocess.fs)
data_analyzer.run(data, remove_erp=True)

# define baseline (example: -1s to 0s)
baseline_start = int(0.0 * preprocess.fs)
baseline_end   = int(1.0 * preprocess.fs)

data_analyzer.apply_baseline(baseline_start, baseline_end)
data_analyzer.apply_spatial_filter(method='laplacian')

# Plot ERD/ERS for first trial (reference-free data)
fig = merge2Signals(
    data_analyzer.alpha[:,:,:],  # Fixed typo: alpha
    data_analyzer.beta[:,:,:],   # Fixed typo: beta
    fs=preprocess.fs,
    label1="Alpha ERD (%) - Laplacian",
    label2="Beta ERS (%) - Laplacian"
)

figs = plot2Signal(
    data_analyzer.alpha,  # Fixed typo
    data_analyzer.beta,   # Fixed typo
    fs=preprocess.fs,
    label1="Alpha ERD (%) - Laplacian",
    label2="Beta ERS (%) - Laplacian",
)

# Plot topographic maps
fig2 = plot_topographic_map(
    data_analyzer.alpha,  # Fixed typo
    fs=preprocess.fs,
    time_points=[0, 1, 2, 3],
    title="Alpha ERD Topography (Laplacian)"
)

plt.show()