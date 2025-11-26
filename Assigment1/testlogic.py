from Coeficient import Coeficient
from Utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HRV import HRV as feature

from Analyze import Respiratory, Vasometric, HeartRate

# Load data
# load_file = pd.read_csv("data/bidmc_07_Signals.csv")
# load_file = pd.read_csv("data/wayan2.csv")
load_file = pd.read_csv("data/karen.csv")
# load_file = pd.read_csv("data/amanda-real.csv")

# Initialize Coeficient
# fs = 120
# fs = 100
fs = 50
factor = 3
fs = fs / factor

coef = Coeficient(fs)
coef.initialize_qj_filter()
HR = HeartRate(fs)
Resp = Respiratory(fs)
Vaso = Vasometric(fs)

selected_signal = load_file.columns[1]
# selected_signal = load_file.columns[2]
signal = downSample(load_file[selected_signal].values, factor)

# == preprocessing ==
mean_signal = np.mean(signal)
signal = signal - mean_signal  # Centering
signal = BPF(signal, 1, 45, fs) # Bandpass Filter 1-40 Hz

# == Analysis ==
# Heart Rate Analysis
signal_Hr, peaks_Hr, BPM = HR.analyze(signal)
rr_intervals = np.diff(peaks_Hr / fs)
print(f"BPM: {BPM}")

# Apply DWT
J_Resp = 5
J_vaso = 8

# Respiratory Analysis
signal_DWT = coef.applying(signal, specific_j=J_Resp)
resp_data , resp_peaks, BrPM = Resp.analyze(signal_DWT[J_Resp])
freq, magnitude, peak_freq, peak_mag = Resp.get_freq()
resp_duration = len(signal_DWT[J_Resp]) / fs  # in seconds
print(f"Respiratory frequency: {peak_freq} Hz")
print(f"BrPM: {BrPM}")

# Vasometric Analysis
signal_dwtvaso = coef.applying(signal, specific_j=J_vaso)
vaso_freq, vaso_mag, peak_vaso, vaso_peak_mag = Vaso.analyze(signal_dwtvaso[J_vaso])
print(f"Vasometric Peak Frequency: {peak_vaso} Hz")

# Compute HRV features
hrv = feature(rr_intervals)
hrv.print_time_features()
hrv.print_nonlinear_features()
hrv.print_frequency_features()

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(signal_Hr, label='Heart Rate Signal')
plt.plot(peaks_Hr, signal_Hr[peaks_Hr], "x", label='Detected Peaks')
plt.title(f'Heart Rate Signal - BPM: {BPM:.2f}')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.subplot(3,1,2)
plt.plot(signal_DWT[J_Resp], label='Respiratory Signal (DWT Level 7)')
plt.plot(resp_peaks, signal_DWT[J_Resp][resp_peaks], "x", label='Detected Peaks')
plt.title(f'Respiratory Signal - Peak Freq: {peak_freq:.2f} Hz')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.subplot(3,1,3)
plt.plot(signal_dwtvaso[J_vaso], label='Vasometric Signal (DWT Level 8)')
plt.title(f'Vasometric Signal - Peak Freq: {peak_vaso:.2f} Hz')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
