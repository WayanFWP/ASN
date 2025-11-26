import numpy as np
import pandas as pd
from Utils import *
from plot import *

namefile = "a0007"
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

selected_peaks = select_r_peak_for_segmentation(r_peaks, time, bp_ecg, PCG_data.values)

if selected_peaks is None:
    print("No R-peak selected")
    exit()

pcg_segments = convert_r_peak_to_segment(selected_peaks, PCG_data, time, fs)

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

s1, s2 = 0.7, 0.05

result = detect_s1_s2(CWT_PCG[0]['coefficients'],
                      CWT_PCG[0]['frequencies'],
                      CWT_PCG[0]['time'],
                      thresh_s1=s1,
                      thresh_s2=s2)

if result is None:
    print("No heart sounds detected")
    exit()

if result[1] is not None:  # Both S1 and S2 detected
    (s1_t, s1_f, s1_mask), (s2_t, s2_f, s2_mask), e = result
    
    seg = pcg_segments[0]
    offset = seg['start_time']
    
    s1_t_global = s1_t + offset
    s2_t_global = s2_t + offset
    
    plot_scalogram_with_s1_s2(CWT_PCG[0]['coefficients'], CWT_PCG[0]['frequencies'], 
                              CWT_PCG[0]['time'], (s1_t, s1_f, s1_mask), 
                              (s2_t, s2_f, s2_mask), e, offset=offset)
else:  # Only S1 detected
    (s1_t, s1_f, s1_mask), _, e = result
    
    seg = pcg_segments[0]
    offset = seg['start_time']
    
    print("S2 not detected")
    
    # Plot with only S1
    plot_scalogram_with_s1_only(CWT_PCG[0]['coefficients'], CWT_PCG[0]['frequencies'], 
                                CWT_PCG[0]['time'], (s1_t, s1_f, s1_mask), e, offset=offset)

window_size = 512
hop_size = 64
scale = False

# STFT Analysis
STFT_PCG = []
for i, seg_data in enumerate(pcg_segments):
    matrix, frequencies, time_frames = stft(seg_data['segment'], window_size, hop_size, window='triangular')
    if frequencies is not None:
        STFT_PCG.append({
            'frequencies': frequencies,
            'time': time_frames,
            'magnitude': matrix,
            'cycle_index': i
        })

plot_stft_result(STFT_PCG[0]['magnitude'], STFT_PCG[0]['frequencies'], STFT_PCG[0]['time'], fs=fs)
