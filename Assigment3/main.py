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
# plot_signals(filtered, fs, cycles)

filtered["gl_denoised"] = filt.dwt_denoise(filtered["gl"].values)
# plot_dwt(filtered["gl"].values, filtered["gl_denoised"].values, fs, title="DWT Denoising - GL")

filtered["vl_denoised"] = filt.dwt_denoise(filtered["vl"].values)
# plot_dwt(filtqered["vl"].values, filtered["vl_denoised"].values, fs, title="DWT Denoising - VL")

# Step 4: CWT on GL and VL segments
gl_segments = extract_cycles(filtered["gl_denoised"].values, cycles)
vl_segments = extract_cycles(filtered["vl_denoised"].values, cycles)

selected_gl_cycles = gl_segments[:3]  # Analyze first 3 cycles for demonstration
selected_vl_cycles = vl_segments[:3]  # Analyze first 3 cycles for demonstration

power_gl = []
power_vl = []
for i, seg in enumerate(selected_gl_cycles):
    gl_segment = seg["segment"]
    cwt_matrix, frequencies, time_axis = filt.cwt_analysis(gl_segment)
    if cwt_matrix is not None:
        power_gl = cwt_power(cwt_matrix)
        plot_scalogram(cwt_matrix, 1/frequencies, time_axis, title=f"CWT Scalogram - GL Segment {i+1}")

for i, seg in enumerate(selected_vl_cycles):
    vl_segment = seg["segment"]
    cwt_matrix, frequencies, time_axis = filt.cwt_analysis(vl_segment)
    if cwt_matrix is not None:
        power_vl = cwt_power(cwt_matrix)
        plot_scalogram(cwt_matrix, 1/frequencies, time_axis, title=f"CWT Scalogram - VL Segment {i+1}")
        
# Step 5: 1% thresholding for getting onset-offset from CWT coefficients
onset = []
offset = []
for cycle_idx, (hs, to, hs_next) in enumerate(cycles):
    segment = filtered["gl_denoised"].values[hs:hs_next]
    
    # Compute CWT power for current cycle
    cwt_matrix_gl, frequencies_gl, time_axis_gl = filt.cwt_analysis(segment)
    current_power_gl = None
    current_power_vl = None
    
    if cwt_matrix_gl is not None:
        current_power_gl = cwt_power(cwt_matrix_gl)
    
    vl_segment = filtered["vl_denoised"].values[hs:hs_next]
    cwt_matrix_vl, frequencies_vl, time_axis_vl = filt.cwt_analysis(vl_segment)
    if cwt_matrix_vl is not None:
        current_power_vl = cwt_power(cwt_matrix_vl)
    
    if current_power_gl is not None and len(current_power_gl) > 0:
        onset_idx, offset_idx = detect_onset_offset(current_power_gl, fs, threshold_ratio=0.01)
        if onset_idx is not None and offset_idx is not None:
            onset.append(hs + onset_idx)
            offset.append(hs + offset_idx)
        else:
            onset.append(None)
            offset.append(None)
    elif current_power_vl is not None and len(current_power_vl) > 0:
        onset_idx, offset_idx = detect_onset_offset(current_power_vl, fs, threshold_ratio=0.01)
        if onset_idx is not None and offset_idx is not None:
            onset.append(hs + onset_idx)
            offset.append(hs + offset_idx)
        else:
            onset.append(None)
            offset.append(None)
    else:
        onset.append(None)
        offset.append(None)

# Plot all cycles with onset/offset detection in one figure
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle("Onset/Offset Detection - All Cycles", fontsize=16, fontweight='bold')

# Time axis for entire signal
t = np.arange(len(filtered["foot_switch"])) / fs

# Foot switch subplot
axes[0].plot(t, filtered["foot_switch"], 'r', label='Foot Switch')
for i, (on, off) in enumerate(zip(onset, offset)):
    if on is not None and off is not None:
        axes[0].axvline(t[on], color='green', linestyle='--', alpha=0.7, 
                       label='Onset' if i == 0 else None)
        axes[0].axvline(t[off], color='red', linestyle='--', alpha=0.7,
                       label='Offset' if i == 0 else None)
axes[0].set_title('Foot Switch')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(alpha=0.3)

# GL subplot
axes[1].plot(t, filtered["gl_denoised"], 'orange', label='GL EMG')
for i, (on, off) in enumerate(zip(onset, offset)):
    if on is not None and off is not None:
        axes[1].axvline(t[on], color='green', linestyle='--', alpha=0.7,
                       label='Onset' if i == 0 else None)
        axes[1].axvline(t[off], color='red', linestyle='--', alpha=0.7,
                       label='Offset' if i == 0 else None)
axes[1].set_title('GL - Gastrocnemius Lateralis')
axes[1].set_ylabel('EMG (mV)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# VL subplot
axes[2].plot(t, filtered["vl_denoised"], 'brown', label='VL EMG')
for i, (on, off) in enumerate(zip(onset, offset)):
    if on is not None and off is not None:
        axes[2].axvline(t[on], color='green', linestyle='--', alpha=0.7,
                       label='Onset' if i == 0 else None)
        axes[2].axvline(t[off], color='red', linestyle='--', alpha=0.7,
                       label='Offset' if i == 0 else None)
axes[2].set_title('VL - Vastus Lateralis')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('EMG (mV)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()