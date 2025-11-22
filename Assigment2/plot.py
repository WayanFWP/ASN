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

def plot_stft_result(stft_result, idx=0):
    """Plot STFT spectrogram"""
    plt.figure(figsize=(10, 6))
    times = stft_result[idx]['time']
    freqs = stft_result[idx]['frequencies']
    mag = stft_result[idx]['magnitude']
    
    # Convert magnitude to dB scale for better visualization
    mag_db = 20 * np.log10(mag + 1e-12)  # Add small value to avoid log(0)
    
    plt.pcolormesh(times, freqs, mag_db, shading='auto', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'STFT Spectrogram - Cardiac Cycle {idx+1}')
    plt.ylim([0, min(500, freqs.max())])  # Focus on heart sound frequencies
    plt.tight_layout()
    plt.show()  
      
def plot_analysis_results(cwt_results, stft_results, idx=0):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plot_cwt_result(cwt_results, idx)
    plot_stft_result(stft_results, idx)
    
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
            label = f'Peak {len(peaks)}'
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
    r_peak_end = selected_peaks[1] * 2
    
    # Extract PCG segment between the two selected R-peaks
    pcg_segment = pcg_data.values[r_peak_start:r_peak_end]
    time_segment = time_data.values[r_peak_start:r_peak_end]
    
    print(f"Selected interval: samples {r_peak_start} to {r_peak_end}")
    print(f"Segment duration: {len(pcg_segment)/fs:.3f} seconds")
    
    # Plot the selected segment
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data.values, pcg_data.values, 'k-', alpha=0.7, label='Full PCG')
    plt.plot(time_segment, pcg_segment, 'r-', linewidth=2, label='Selected PCG Segment')
    plt.axvline(time_data.values[r_peak_start], color='r', linestyle='--', label='Start R-peak')
    plt.axvline(time_data.values[r_peak_end], color='g', linestyle='--', label='End R-peak')
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
        'start_r_peak': r_peak_start,
        'end_r_peak': r_peak_end,
        'duration_samples': len(pcg_segment),
        'duration_seconds': len(pcg_segment)/fs
    }]
    
def plot_scalogram_with_s1_s2(coeff, scales, times, s1, s2, E):
    (t1, f1, S1_mask) = s1
    (t2, f2, S2_mask) = s2

    plt.figure(figsize=(14, 6))

    plt.pcolormesh(times, scales, E, shading="auto", cmap="viridis")
    plt.colorbar(label="Energy")

    # MASK overlay
    plt.contour(times, scales, S1_mask, levels=[0.5], colors="red", linewidths=2)
    plt.contour(times, scales, S2_mask, levels=[0.5], colors="cyan", linewidths=2)

    # Plot CoG points (pakai scale = 1/frequency)
    plt.scatter(t1, 1/f1, color="red", s=100, label="S1 CoG")
    plt.scatter(t2, 1/f2, color="cyan", s=100, label="S2 CoG")

    plt.xlabel("Time (s)")
    plt.ylabel("Scale (1/Frequency)")
    plt.title("CWT Scalogram with S1 & S2 CoG")
    plt.legend()
    plt.show()
