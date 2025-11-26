import matplotlib.pyplot as plt
import numpy as np  

def plot_signals(x, ecg_signal, r_peaks, pcg_signal):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(x[r_peaks], ecg_signal[r_peaks], 'ro', label='Detected R-peaks')
    axs[0].plot(x, ecg_signal, label='Bandpass Filtered ECG')
    axs[0].legend()

    axs[1].plot(x, pcg_signal, label='Raw PCG')
    axs[1].set_title('ECG Signal with Detected R-Peaks')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
# Plot first few results
def plot_cwt_result(coeff, scales, times):
    power = np.abs(coeff)**2
    power /= power.max()
    scales = 1/scales  # Convert scales to pseudo-frequencies

    plt.figure(figsize=(12,6))
    plt.pcolormesh(times, scales, power, shading='auto', cmap='jet')
    plt.xlabel("Time (s)")
    plt.ylabel("Scale (1/Frequency)")
    plt.title("CWT Scalogram (Scale Domain)")
    plt.colorbar(label="Normalized Power")
    plt.show()

def plot_stft_result(stft_matrix, frequencies, time_frames, fs=1, db_scale=True):
    magnitude = np.abs(stft_matrix)
    
    if db_scale:
        magnitude = 20 * np.log10(magnitude + 1e-10)
    
    # Only plot positive frequencies
    positive_freq_idx = frequencies >= 0
    magnitude_positive = magnitude[positive_freq_idx, :]
    frequencies_positive = frequencies[positive_freq_idx] * fs
    
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_positive, aspect='auto', origin='lower',
               extent=[time_frames[0]/fs, time_frames[-1]/fs, 
                       frequencies_positive[0], frequencies_positive[-1]])
    plt.colorbar(label='Magnitude (dB)' if db_scale else 'Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('STFT Spectrogram')
    plt.tight_layout()
    plt.show()
    
def select_peaks_interactively(signal, time):
    """Click on the plot to select R-peaks"""
    plt.figure(figsize=(15, 6))
    plt.plot(time, signal)
    plt.title('Click on exactly 2 R-peaks to define interval, then close the window')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    peaks = []
    
    def onclick(event):
        if event.inaxes and len(peaks) < 2:
            # Find closest sample to clicked time
            click_time = event.xdata
            closest_idx = np.argmin(np.abs(time - click_time))
            peaks.append(closest_idx)
            
            color = 'ro' if len(peaks) == 1 else 'go'
            label = f'point {len(peaks)}'
            plt.plot(time[closest_idx], signal[closest_idx], color, markersize=8, label=label)
            plt.legend()
            plt.draw()
            print(f"Selected peak {len(peaks)} at time {click_time:.3f}s, sample {closest_idx}")
            
            if len(peaks) == 2:
                # Draw interval
                start_time = time[peaks[0]]
                end_time = time[peaks[1]]
                plt.axvspan(start_time, end_time, alpha=0.3, color='yellow', label='Selected Interval')
                plt.legend()
                plt.draw()
                print(f"Interval selected: {start_time:.3f}s to {end_time:.3f}s")
    
    plt.connect('button_press_event', onclick)
    plt.show()
    
    if len(peaks) == 2:
        return np.array(sorted(peaks))  # Ensure chronological order
    else:
        print("Warning: Less than 2 peaks selected")
        return np.array([])
    
def convert_peaks_to_intervals(selected_peaks, pcg_data, time_data, fs):
    """
    Convert selected peaks to PCG intervals and handle all the analysis logic
    """
    if len(selected_peaks) != 2:
        print(f"Error: Exactly 2 R-peaks must be selected. You selected {len(selected_peaks)} peaks.")
        return []
    
    # Convert selected peaks back to original sampling rate indices
    r_peak_start = selected_peaks[0] * 2  # Convert back from downsampled indices
    r_peak_end   = selected_peaks[1] * 2
    
    # Extract PCG segment between the two selected R-peaks
    pcg_segment = pcg_data.values[r_peak_start:r_peak_end]
    time_segment = time_data.values[r_peak_start:r_peak_end]

    # === PATCH: Add global times ===
    start_time_global = float(time_data.values[r_peak_start])
    end_time_global   = float(time_data.values[r_peak_end])
    # ================================

    print(f"Selected interval: samples {r_peak_start} to {r_peak_end}")
    print(f"Segment duration: {len(pcg_segment)/fs:.3f} seconds")

    # Plot the selected segment
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data.values, pcg_data.values, 'k-', alpha=0.7, label='Full PCG')
    plt.plot(time_segment, pcg_segment, 'r-', linewidth=2, label='Selected PCG Segment')
    plt.axvline(start_time_global, color='r', linestyle='--', label='Start R-peak')
    plt.axvline(end_time_global, color='g', linestyle='--', label='End R-peak')
    plt.ylabel('PCG Amplitude')
    plt.legend()
    plt.title('Selected PCG Interval')

    plt.subplot(2, 1, 2)
    plt.plot(time_segment, pcg_segment, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('PCG Amplitude')
    plt.title('Zoomed Selected Segment')
    plt.tight_layout()
    plt.show()

    return [{
        'segment': pcg_segment,
        'time': time_segment,
        'start_index': r_peak_start,
        'end_index': r_peak_end,
        'start_time': start_time_global,    # <-- PATCH
        'end_time': end_time_global,        # <-- PATCH
        'duration_samples': len(pcg_segment),
        'duration_seconds': len(pcg_segment)/fs
    }]

def select_r_peak_for_segmentation(r_peaks, time_data, ecg_signal, pcg_signal):
    """
    Display detected R-peaks and let user select one R-peak for ±500ms segmentation
    """
    if len(r_peaks) < 1:
        print("Error: Need at least 1 R-peak")
        return None
    
    # Display available R-peaks
    print("\nAvailable R-peaks:")
    print("Index | Time | Sample")
    print("-" * 30)
    for i, peak_idx in enumerate(r_peaks):
        peak_time = time_data.values[peak_idx]
        print(f"{i:5d} | {peak_time:6.3f}s | {peak_idx:8d}")
    
    # Plot all R-peaks for visual selection
    plt.figure(figsize=(15, 10))
    
    # Plot ECG with all R-peaks
    plt.subplot(2, 1, 1)
    plt.plot(time_data.values, ecg_signal, 'b-', alpha=0.7, label='Filtered ECG')
    plt.plot(time_data.values[r_peaks], ecg_signal[r_peaks], 'ro', markersize=8, label='R-peaks')
    
    # Highlight each R-peak with different colors and show ±500ms windows
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    fs = 2000  # sampling frequency
    window_ms = 500
    window_samples = int(window_ms * fs / 1000)  # 500ms in samples
    
    for i, peak_idx in enumerate(r_peaks):  # Show all detected R-peaks
        color = colors[i % len(colors)]
        peak_time = time_data.values[peak_idx]
        
        # Show ±500ms window
        start_time = peak_time - window_ms/1000
        end_time = peak_time + window_ms/1000
        
        plt.axvspan(start_time, end_time, alpha=0.3, color=color)
        plt.axvline(peak_time, color=color, linestyle='-', linewidth=2)
        
        # Add R-peak index text
        plt.text(peak_time, max(ecg_signal) * 0.9, str(i), 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Amplitude')
    plt.title('ECG with R-peaks and ±500ms Windows')
    plt.legend()  # Removed the bbox_to_anchor to keep it inside
    plt.grid(True)
    
    # Plot PCG signal with R-peak windows
    plt.subplot(2, 1, 2)
    plt.plot(time_data.values, pcg_signal, 'k-', alpha=0.7, label='PCG Signal')
    
    # Highlight ±500ms windows in PCG
    for i, peak_idx in enumerate(r_peaks):
        color = colors[i % len(colors)]
        peak_time = time_data.values[peak_idx]
        
        start_time = peak_time - window_ms/1000
        end_time = peak_time + window_ms/1000
        
        plt.axvspan(start_time, end_time, alpha=0.3, color=color)
        plt.axvline(peak_time, color=color, linestyle='-', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('PCG Amplitude')
    plt.title('PCG Signal with R-peak ±500ms Windows')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Get user selection
    while True:
        try:
            selected_idx = int(input(f"\nSelect R-peak index (0-{len(r_peaks)-1}): "))
            if 0 <= selected_idx < len(r_peaks):
                selected_peak = r_peaks[selected_idx]
                peak_time = time_data.values[selected_peak]
                
                print(f"\nSelected R-peak {selected_idx}:")
                print(f"  Time: {peak_time:.3f}s")
                print(f"  Sample: {selected_peak}")
                print(f"  Segment: ±500ms around R-peak")
                
                return selected_peak
            else:
                print(f"Please enter a number between 0 and {len(r_peaks)-1}")
        except ValueError:
            print("Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None

def convert_r_peak_to_segment(selected_r_peak, pcg_data, time_data, fs, window_ms=500):
    """
    Convert single R-peak to ±500ms PCG segment
    """
    window_samples = int(window_ms * fs / 1000)  # Convert ms to samples
    
    # Calculate segment boundaries
    start_idx = max(0, selected_r_peak - window_samples)
    end_idx = min(len(pcg_data), selected_r_peak + window_samples)
    
    # Extract PCG segment
    pcg_segment = pcg_data.values[start_idx:end_idx]
    time_segment = time_data.values[start_idx:end_idx]
    
    peak_time = time_data.values[selected_r_peak]
    start_time_global = float(time_data.values[start_idx])
    end_time_global = float(time_data.values[end_idx])
    
    actual_duration_ms = (end_idx - start_idx) / fs * 1000
    
    print(f"R-peak segment created:")
    print(f"  R-peak at: {peak_time:.3f}s (sample {selected_r_peak})")
    print(f"  Segment: samples {start_idx} to {end_idx}")
    print(f"  Duration: {actual_duration_ms:.1f}ms")
    print(f"  Time range: {start_time_global:.3f}s to {end_time_global:.3f}s")
    
    # Plot the selected segment
    plt.figure(figsize=(12, 8))
    
    # Full signal with selected segment
    plt.subplot(3, 1, 1)
    plt.plot(time_data.values, pcg_data.values, 'k-', alpha=0.7, label='Full PCG')
    plt.plot(time_segment, pcg_segment, 'r-', linewidth=2, label='Selected Segment')
    plt.axvline(peak_time, color='b', linestyle='-', linewidth=2, label='R-peak')
    plt.axvline(start_time_global, color='r', linestyle='--', alpha=0.7, label='Segment Start')
    plt.axvline(end_time_global, color='g', linestyle='--', alpha=0.7, label='Segment End')
    plt.ylabel('PCG Amplitude')
    plt.legend()
    plt.title('PCG Signal with Selected ±500ms Segment')
    plt.grid(True)
    
    # Zoomed segment in absolute time
    plt.subplot(3, 1, 2)
    plt.plot(time_segment, pcg_segment, 'b-', linewidth=2)
    plt.axvline(peak_time, color='r', linestyle='-', linewidth=2, label='R-peak')
    plt.xlabel('Time (s)')
    plt.ylabel('PCG Amplitude')
    plt.title('Selected Segment (Absolute Time)')
    plt.legend()
    plt.grid(True)
    
    # Segment centered on R-peak (relative time)
    plt.subplot(3, 1, 3)
    relative_time = (time_segment - peak_time) * 1000  # Convert to ms, centered on R-peak
    plt.plot(relative_time, pcg_segment, 'g-', linewidth=2)
    plt.axvline(0, color='r', linestyle='-', linewidth=2, label='R-peak')
    plt.axvspan(-200, 100, alpha=0.3, color='yellow', label='Expected S1 region')
    plt.axvspan(100, 400, alpha=0.3, color='cyan', label='Expected S2 region')
    plt.xlabel('Time from R-peak (ms)')
    plt.ylabel('PCG Amplitude')
    plt.title('Segment Relative to R-peak (±500ms)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return [{
        'segment': pcg_segment,
        'time': time_segment,
        'r_peak_index': selected_r_peak,
        'start_index': start_idx,
        'end_index': end_idx,
        'start_time': start_time_global,
        'end_time': end_time_global,
        'r_peak_time': peak_time,
        'duration_samples': len(pcg_segment),
        'duration_seconds': len(pcg_segment)/fs,
        'duration_ms': actual_duration_ms
    }]

def plot_scalogram_with_s1_s2(coeff, scales, times, s1, s2, E, offset=0):
    (t1, scale_plot_1, S1_mask) = s1  # Now already in plotting coordinates
    (t2, scale_plot_2, S2_mask) = s2  # Now already in plotting coordinates

    plt.figure(figsize=(14, 6))
    
    scales_plot = 1/scales  # Convert scales to pseudo-frequencies for plotting

    # Normalize energy to 0-1 range
    E_normalized = (E - E.min()) / (E.max() - E.min())

    plt.pcolormesh(times, scales_plot, E_normalized, shading="auto", cmap="plasma")
    plt.colorbar(label="Normalized Energy")

    # MASK overlay
    plt.contour(times, scales_plot, S1_mask, levels=[0.5], colors="red", linewidths=2)
    plt.contour(times, scales_plot, S2_mask, levels=[0.5], colors="cyan", linewidths=2)

    # Plot CoG points - coordinates already match plotting system
    f1_actual = 1/scale_plot_1  # Convert to frequency for display
    f2_actual = 1/scale_plot_2
    
    plt.scatter(t1, scale_plot_1, color="red", s=100, label=f"S1 CoG {t1+offset:.3f}s, freq {1/f1_actual:.2f}Hz")
    plt.scatter(t2, scale_plot_2, color="cyan", s=100, label=f"S2 CoG {t2+offset:.3f}s, freq {1/f2_actual:.2f}Hz")

    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    plt.title("CWT Scalogram with S1 & S2 CoG")
    plt.legend()
    plt.show()

def plot_scalogram_with_s1_only(coeff, scales, times, s1, E, offset=0):
    (t1, scale_plot_1, S1_mask) = s1  # Now already in plotting coordinates

    plt.figure(figsize=(14, 6))
    
    scales_plot = 1/scales  # Convert scales to pseudo-frequencies for plotting

    # Normalize energy to 0-1 range
    E_normalized = (E - E.min()) / (E.max() - E.min())

    plt.pcolormesh(times, scales_plot, E_normalized, shading="auto", cmap="plasma")
    plt.colorbar(label="Normalized Energy")

    # MASK overlay for S1 only
    plt.contour(times, scales_plot, S1_mask, levels=[0.5], colors="red", linewidths=2)

    # Plot CoG point - coordinates already match
    f1_actual = 1/scale_plot_1
    plt.scatter(t1, scale_plot_1, color="red", s=100, label=f"S1 CoG {t1+offset:.3f}s, freq {1/f1_actual:.2f}Hz")

    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    plt.title("CWT Scalogram with S1 CoG (S2 not detected)")
    plt.legend()
    plt.show()