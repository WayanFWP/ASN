from Utils import *
from preproceed import *
from Plot import *
import matplotlib.pyplot as plt
import numpy as np

preprocess = dataLoader("B0101T")
data, labels = preprocess.X, preprocess.y
print(f"Data shape: {data.shape}")  # (n_trials, n_channels, n_samples)

baseline_data = np.zeros_like(data)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        baseline_data[i, j, :] = BPF(data[i, j, :], fs=preprocess.fs)
print("BandPass Filter Applied")

# ===== ERD/ERS ANALYSIS =====
# Squaring (Power)
squared_data = squaring(baseline_data)
print("Squaring Applied")

# Moving Average (Smoothing)
averaged_data = np.zeros_like(squared_data)
for i in range(squared_data.shape[0]):
    for j in range(squared_data.shape[1]):
        averaged_data[i, j, :] = avaragingOverN(squared_data[i, j, :], N=10)
print("Averaging Over N Applied")