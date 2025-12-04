from Filter import Filter
from utils import *
from plot import *
import pandas as pd

fs = 2000
filt = Filter(fs)
namefile = "S01"

# Load
data = pd.read_csv(f"data/{namefile}_extracted.csv")

# Step 1: filter EMG
filtered = data.copy()
for col in data.columns:
    if col != "foot_switch":
        filtered[col] = filt.BPF(data[col].values, 20, 450)

# Step 2: detect gait cycle events
cycles = detect_cycle(filtered["foot_switch"])

# Step 3: plot
plot_signals(filtered, fs, cycles)

filtered["gl_denoised"] = filt.dwt_denoise(filtered["gl"].values)
filtered["vl_denoised"] = filt.dwt_denoise(filtered["vl"].values)

# Step 4: Extract toe-off events for segmentation
toe_off_events = extract_toe_off_events(filtered["foot_switch"].values)
print(f"Found {len(toe_off_events)} toe-off events at indices: {toe_off_events}")

# Step 5: Segment gait cycles based on toe-off events
emg_signals = {
    'gl': filtered['gl'],
    'vl': filtered['vl'], 
    'gl_denoised': filtered['gl_denoised'],
    'vl_denoised': filtered['vl_denoised']
}

gait_segments = segment_gait(emg_signals, toe_off_events)
print(f"Created {len(gait_segments)} gait cycle segments")

if len(gait_segments) > 0:
    _segmented = input(f"Selected segment: ")
    segmented = gait_segments[int(_segmented)]
    cwt_gl_coeffs, freqs_gl, time_axis_gl = filt.cwt_analysis(segmented['gl'].values, percentage_keep=0.01)    
    cwt_vl_coeffs, freqs_vl, time_axis_vl = filt.cwt_analysis(segmented['vl'].values, percentage_keep=0.1)
    
    plot_scalogram(cwt_gl_coeffs, freqs_gl, time_axis_gl, title=f"CWT - GL Segment {int(_segmented)}")
    plot_scalogram(cwt_vl_coeffs, freqs_vl, time_axis_vl, title=f"CWT - VL Segment {int(_segmented)}")
    
# step 6: turn the cwt analysis data into a onset and offset based on the thresholding method described in the assignment sheet

