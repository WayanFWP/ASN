import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils import *
from plot import *

namefile = "a0004"
df = pd.read_csv('dat/'+namefile+'.csv')
print(df.shape)
time = df.iloc[:, 0]
PCG_data = df.iloc[:, 1]
ECG_data = df.iloc[:, 2]

dt = np.diff(time)

fs = 2000  # Sampling frequency in Hz

print("select R-peaks interactively...")
bp_ecg, r_peaks = pan_tompkins(ECG_data.values, fs)
print(f"Intervals Peaks {len(r_peaks)}")

# Check First glance of results by plotting
plot_signals(time, bp_ecg, r_peaks, PCG_data.values)

data_r ,_ = pan_tompkins(ECG_data.values, fs)
selected_peaks = select_peaks_interactively(data_r[::2], time.values[::2])

pcg_segments = convert_peaks_to_intervals(selected_peaks, PCG_data, time, fs)

# CWT Analysis
CWT_PCG = []
for i, seg_data in enumerate(pcg_segments):
    coefficients, frequencies, time_axis = cwt_analysis(seg_data['segment'], fs)
    if coefficients is not None:
        CWT_PCG.append({
            'coefficients': coefficients,
            'frequencies': frequencies,
            'time': time_axis,
            'r_peak_index': seg_data['start_index'],
            'cycle_index': i
        })
        
# STFT Analysis
STFT_PCG = []
for i, seg_data in enumerate(pcg_segments):
    freqs, times, spectrogram = stft_analysis(seg_data['segment'], fs, 100, 10)
    if freqs is not None:
        STFT_PCG.append({
            'frequencies': freqs,
            'time': times,
            'magnitude': spectrogram,
            'cycle_index': i
        })
# Plot results
plot_stft_result(STFT_PCG, idx=0)

s1, s2 = 0.5, 0.08

result = detect_s1_s2(CWT_PCG[0]['coefficients'],
                      CWT_PCG[0]['frequencies'],
                      CWT_PCG[0]['time'],
                      thresh_s1=s1,
                      thresh_s2=s2)

if result is None:
    print("No S1/S2 detected")
    exit()

(s1_t, s1_f, s1_mask), (s2_t, s2_f, s2_mask), e = result

seg = pcg_segments[0]
offset = seg['start_time']

s1_t_global = s1_t + offset
s2_t_global = s2_t + offset

print(f"S1 at time {s1_t_global:.3f}s, frequency {s1_f:.2f}Hz")
print(f"S2 at time {s2_t_global:.3f}s, frequency {s2_f:.2f}Hz")
plot_scalogram_with_s1_s2(CWT_PCG[0]['coefficients'], CWT_PCG[0]['frequencies'], CWT_PCG[0]['time'], (s1_t, s1_f, s1_mask), (s2_t, s2_f, s2_mask), e, offset=offset)