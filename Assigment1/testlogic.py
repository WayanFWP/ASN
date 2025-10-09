from Coeficient import Coeficient
from Utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import find_peaks
from FindPeak import find_peaks

# Initialize Coeficient
fs = 125
factor = 3
fs = fs / factor

coef = Coeficient(fs)
coef.initialize_qj_filter()

load_file = pd.read_csv("../data/bidmc_01_Signals.csv")

# Load data
print("Available columns:", load_file.columns.tolist())
selected_signal = "PLETH" if "PLETH" in load_file.columns else load_file.columns[2]
time = downSample(load_file.iloc[:, 0].values, factor)  # Use first column for time
signal = downSample(load_file[selected_signal].values, factor)
data = pd.DataFrame({'Time': time, 'Signal': signal})

mean_signal = np.mean(data['Signal'].values)
data['Signal'] = data['Signal'].values - mean_signal  # Centering
# Apply DWT1-8
selected_j = 6
signal_DWT = coef.applying(data['Signal'].values, specific_j=selected_j)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(data['Signal'], label="Raw Data", color="red")
plt.plot(signal_DWT[selected_j], label=f"Dwt {selected_j}", color="blue")
plt.ylabel("Amp")
plt.legend()

time_diff = np.arange(len(signal_DWT[selected_j]))/fs

peaks = find_peaks(signal_DWT[selected_j], height=np.mean(signal_DWT[selected_j])*1.5, distance =fs*0.5)
rr_interval = np.diff(time_diff[peaks])
BrPM = 60 / np.mean(rr_interval) 
print(len(rr_interval))
print(BrPM)

# Mark peaks with larger red circles
plt.scatter(peaks, signal_DWT[selected_j][peaks], color='red', s=100, 
           marker='o', edgecolors='black', linewidth=2, label='Peaks', zorder=5)
plt.ylabel("Amplitude")
plt.xlabel("Sample Index")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

hrv_result= hrv_time_domain(rr_interval)
print("HRV Time Domain Metrics:")
for key, value in hrv_result.items():
    print(f"{key}: {value}")
    
hrv_freq = hrv_frequency_domain(rr_interval)
print("\nHRV Frequency Domain Metrics:")
for key, value in hrv_freq.items():
    print(f"{key}: {value}")
    
hrv_nonlin = hrv_nonlinear(rr_interval, show_plot=False)
print("\nHRV Non-Linear Metrics:")
for key, value in hrv_nonlin.items():
    print(f"{key}: {value}")