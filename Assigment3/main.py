from Filter import Filter
from utils import *
from plot import *
import pandas as pd
import numpy as np

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

print(f"{'='*10}Q value for DWT filtering can use 1-8{'='*10}")
Q = int(input("Enter Q value for DWT filtering (suggested 4): "))

gl_dwt_result = filt.dwt.applying(filtered['gl'].values, specific_j=Q)
vl_dwt_result = filt.dwt.applying(filtered['vl'].values, specific_j=Q)

filtered['gl'] = gl_dwt_result[Q]  # Extract the array for j=Q
filtered['vl'] = vl_dwt_result[Q]  # Extract the array for j=Q

# Step 3: Extract toe-off events for segmentation
toe_off_events = extract_toe_off_events(filtered["foot_switch"].values)
print(f"Found {len(toe_off_events)} toe-off events at indices: {toe_off_events}")

# Step 4: Segment gait cycles based on toe-off events
emg_signals = {
    'gl': filtered['gl'],
    'vl': filtered['vl']
}

gait_segments = segment_gait(emg_signals, toe_off_events)
print(f"Created {len(gait_segments)} gait cycle segments")

print("\n=== FULL DATASET ANALYSIS ===")

# Step 1: Look at the full dataset
show_full_plot = input("Do you want to see the full dataset plot? (y/n): ").lower() == 'y'
if show_full_plot:
    plot_signals(filtered, fs, cycles)

# Initialize storage for all results
all_gl_activations = []  # [(segment_idx, global_onset, global_offset, duration), ...]
all_vl_activations = []
segment_results = []

# Ask user preferences
process_all = input("Process all segments automatically? (y/n): ").lower() == 'y'
show_individual_cwt = input("Show combined CWT plots for segments? (y/n): ").lower() == 'y' if process_all else True
show_individual_detection = input("Show tabbed detection plots for segments? (y/n): ").lower() == 'y' if process_all else True

if process_all:
    segments_to_process = range(len(gait_segments))
else:
    print(f"Available segments: 0 to {len(gait_segments)-1}")
    segment_input = input("Enter segment numbers (comma-separated) or 'all': ")
    if segment_input.lower() == 'all':
        segments_to_process = range(len(gait_segments))
    else:
        segments_to_process = [int(x.strip()) for x in segment_input.split(',')]

print(f"\nProcessing {len(segments_to_process)} segments...")

# Step 2-4: Process each segment
for i, segment_idx in enumerate(segments_to_process):
    print(f"\n{'='*50}")
    print(f"Processing Segment {segment_idx} ({i+1}/{len(segments_to_process)})")
    print(f"{'='*50}")
    
    segmented = gait_segments[segment_idx]
    
    # Get global time indices for this segment
    global_start_idx = segmented['start_idx']
    global_end_idx = segmented['end_idx']
    
    # CWT Analysis for both muscles
    cwt_gl_coeffs, freqs_gl, time_axis_gl = filt.cwt_analysis(segmented['gl'].values, percentage_keep=0.01)    
    cwt_vl_coeffs, freqs_vl, time_axis_vl = filt.cwt_analysis(segmented['vl'].values, percentage_keep=0.05)
    
    # Step 3: Check CWT results with combined plot
    if show_individual_cwt:
        if segment_idx == int(segment_input):
            print(f"Showing combined CWT plots for Segment {segment_idx}...")
            plot_combined_scalograms(cwt_gl_coeffs, 1/freqs_gl, time_axis_gl,
                                   cwt_vl_coeffs, 1/freqs_vl, time_axis_vl,
                                   segment_idx, "CWT Analysis")
    
    # Step 4: Detect activations    
    # GL Analysis
    gl_activations, gl_energy, gl_threshold = detect_multiple_activations_cwt(
        cwt_gl_coeffs, freqs_gl, fs,
        method='percentile',
        threshold_factor=3.,
        min_duration=0.03,
        min_gap=0.02,
        freq_range=(20, 250),
        merge_window=0.5,
        debug=False  
    )
    
    print(f"  GL - Found {len(gl_activations)} activations")
    
    # Convert local indices to global indices and store
    for local_onset, local_offset in gl_activations:
        global_onset = global_start_idx + local_onset
        global_offset = global_start_idx + local_offset
        duration = (local_offset - local_onset) / fs
        all_gl_activations.append((segment_idx, global_onset, global_offset, duration))
        print(f"    Activation: {local_onset/fs:.3f}s - {local_offset/fs:.3f}s ({duration:.3f}s)")
    
    # VL Analysis
    print(f"Analyzing VL (Vastus Lateralis):")
    
    # Try different methods for VL detection
    vl_detected = False
    test_params = [
        {'method': 'percentile', 'threshold_factor': 0.5, 'freq_range': (21, 250), 'min_duration': 0.02},
        {'method': 'mean', 'threshold_factor': 1.5, 'freq_range': (30, 200), 'min_duration': 0.03},
        {'method': 'teager', 'threshold_factor': 2.0, 'freq_range': (20, 150), 'min_duration': 0.03},
    ]
    
    vl_activations = []
    vl_energy = None
    vl_threshold = None
    
    for params in test_params:
        vl_activations, vl_energy, vl_threshold = detect_multiple_activations_cwt(
            cwt_vl_coeffs, freqs_vl, fs,
            method=params['method'],
            threshold_factor=params['threshold_factor'],
            min_duration=params['min_duration'],
            min_gap=0.01,
            freq_range=params['freq_range'],
            merge_window=0.5,
            debug=False
        )
        
        if len(vl_activations) > 0:
            vl_detected = True
            break
    
    print(f"  VL - Found {len(vl_activations)} activations")
    
    # Convert local indices to global indices and store
    for local_onset, local_offset in vl_activations:
        global_onset = global_start_idx + local_onset
        global_offset = global_start_idx + local_offset
        duration = (local_offset - local_onset) / fs
        all_vl_activations.append((segment_idx, global_onset, global_offset, duration))
        print(f"    Activation: {local_onset/fs:.3f}s - {local_offset/fs:.3f}s ({duration:.3f}s)")
    
    # Store segment results
    segment_results.append({
        'segment_idx': segment_idx,
        'global_start': global_start_idx,
        'global_end': global_end_idx,
        'gl_activations': len(gl_activations),
        'vl_activations': len(vl_activations),
        'gl_local': gl_activations,
        'vl_local': vl_activations
    })
    
    # Show tabbed detection plots if requested
    if show_individual_detection and (len(gl_activations) > 0 or len(vl_activations) > 0):
        if segment_idx == int(segment_input): 
            time_segment = np.arange(len(segmented['gl'])) / fs
            
            # Create tabbed plot for both GL and VL
            plot_tabbed_detection_debug(
                time_segment, segmented['gl'].values, gl_energy, gl_threshold, gl_activations,
                time_segment, segmented['vl'].values, vl_energy if vl_energy is not None else np.zeros_like(time_segment), 
                vl_threshold if vl_threshold is not None else 0, vl_activations,
                segment_idx, "EMG Detection Analysis"
            )

# Step 5: Final Results and Visualization
print(f"\n{'='*60}")
print(f"FINAL RESULTS ACROSS ENTIRE DATASET")
print(f"{'='*60}")

print(f"\nSUMMARY:")
print(f"Total segments processed: {len(segments_to_process)}")
print(f"Total GL activations found: {len(all_gl_activations)}")
print(f"Total VL activations found: {len(all_vl_activations)}")

# Create comprehensive final plot
print(f"\nGenerating comprehensive final analysis plot...")

# Prepare time axis for full dataset
full_time = np.arange(len(filtered)) / fs

# Use the new comprehensive plotting function
if process_all:
    plot_comprehensive_analysis(full_time, filtered, cycles, segments_to_process, 
                            gait_segments, all_gl_activations, all_vl_activations)

# Additional Analysis
print(f"\n{'='*60}")
print(f"STATISTICAL ANALYSIS")
print(f"{'='*60}")

if len(all_gl_activations) > 0:
    gl_durations = [duration for _, _, _, duration in all_gl_activations]
    print(f"\nGL Statistics:")
    print(f"  Average activation duration: {np.mean(gl_durations):.3f} ± {np.std(gl_durations):.3f} s")
    print(f"  Min duration: {np.min(gl_durations):.3f} s")
    print(f"  Max duration: {np.max(gl_durations):.3f} s")

if len(all_vl_activations) > 0:
    vl_durations = [duration for _, _, _, duration in all_vl_activations]
    print(f"\nVL Statistics:")
    print(f"  Average activation duration: {np.mean(vl_durations):.3f} ± {np.std(vl_durations):.3f} s")
    print(f"  Min duration: {np.min(vl_durations):.3f} s")
    print(f"  Max duration: {np.max(vl_durations):.3f} s")

# Activation rate per segment
print(f"\nActivation Rate Analysis:")
total_segments = len(segments_to_process)
gl_active_segments = len(set(seg_idx for seg_idx, _, _, _ in all_gl_activations))
vl_active_segments = len(set(seg_idx for seg_idx, _, _, _ in all_vl_activations))

print(f"  GL activation rate: {gl_active_segments}/{total_segments} segments ({gl_active_segments/total_segments*100:.1f}%)")
print(f"  VL activation rate: {vl_active_segments}/{total_segments} segments ({vl_active_segments/total_segments*100:.1f}%)")

window_size = 64
N_win = int(fs * window_size / 1000)  # 512 ms window
hop_size = N_win // 2
for i, segment_idx in enumerate(segments_to_process):
    segmented = gait_segments[segment_idx]
    
    matrix_gl, frequencies_gl, time_frames_gl = filt.stft(segmented['gl'], window_size, hop_size, window='triangular', percentage=0.01) 
    matrix_vl, frequencies_vl, time_frames_vl = filt.stft(segmented['vl'], window_size, hop_size, window='triangular', percentage=0.01)       
    if process_all is False:
        plot_stft_result(matrix_gl, frequencies_gl, time_frames_gl, fs)
        plot_stft_result(matrix_vl, frequencies_vl, time_frames_vl, fs)

print(f"\n{'='*60}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*60}")