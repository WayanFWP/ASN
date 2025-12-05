import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def _plot_markers(ax, time, cycles):
    for i, (hs, to, hs_next) in enumerate(cycles):
        # Heel strike
        ax.axvline(time[hs], color='green', linestyle='--', alpha=0.8,
                   linewidth=1.2, label="Heel Strike" if i == 0 else None)

        # Toe off (if available)
        if to is not None:
            ax.axvline(time[to], color='blue', linestyle='--', alpha=0.8,
                       linewidth=1.2, label="Toe Off" if i == 0 else None)


def plot_signals(data, fs, cycles=None):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("Signal Analysis", fontsize=16, fontweight='bold')

    # FOOT SWITCH
    t = np.arange(len(data["foot_switch"])) / fs
    axes[0].plot(t, data["foot_switch"], 'r', linewidth=1)
    axes[0].set_title("Foot Switch")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    if cycles:
        _plot_markers(axes[0], t, cycles)
        axes[0].legend()

    # GL
    t = np.arange(len(data["gl"])) / fs
    axes[1].plot(t, data["gl"], color="orange", linewidth=0.8)
    axes[1].set_title("Gastrocnemius Lateralis")
    axes[1].set_ylabel("EMG (mV)")
    axes[1].grid(alpha=0.3)
    if cycles:
        _plot_markers(axes[1], t, cycles)

    # VL
    t = np.arange(len(data["vl"])) / fs
    axes[2].plot(t, data["vl"], color="brown", linewidth=0.8)
    axes[2].set_title("Vastus Lateralis")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("EMG (mV)")
    axes[2].grid(alpha=0.3)
    if cycles:
        _plot_markers(axes[2], t, cycles)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_spectrogram(f, t, Zxx, title="STFT Spectrogram"):
    magnitude = np.abs(Zxx)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, magnitude, shading='gouraud')
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
def plot_scalogram(cwt_matrix, freqs, time, title="CWT Scalogram"):
    power = np.abs(cwt_matrix)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(time, freqs, power, shading='auto')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Power")
    plt.tight_layout()
    plt.show()

def plot_combined_scalograms(cwt_gl, freqs_gl, time_gl, cwt_vl, freqs_vl, time_vl, 
                           segment_idx, title_prefix="CWT Analysis"):
    """
    Plot GL and VL scalograms in a single window with two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"{title_prefix} - Segment {segment_idx}", fontsize=16, fontweight='bold')
    
    # GL Scalogram
    power_gl = np.abs(cwt_gl)
    im1 = ax1.pcolormesh(time_gl, freqs_gl, power_gl, shading='auto', cmap='viridis')
    ax1.invert_yaxis()
    ax1.set_title("Gastrocnemius Lateralis (GL)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.grid(alpha=0.3)
    plt.colorbar(im1, ax=ax1, label="Power")
    
    # VL Scalogram
    power_vl = np.abs(cwt_vl)
    im2 = ax2.pcolormesh(time_vl, freqs_vl, power_vl, shading='auto', cmap='plasma')
    ax2.invert_yaxis()
    ax2.set_title("Vastus Lateralis (VL)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.3)
    plt.colorbar(im2, ax=ax2, label="Power")
    
    plt.tight_layout()
    plt.show()

class TabbedDetectionPlot:
    """
    Create a tabbed interface for GL and VL detection plots using matplotlib widgets.
    """
    def __init__(self, time_gl, signal_gl, energy_gl, threshold_gl, activations_gl,
                 time_vl, signal_vl, energy_vl, threshold_vl, activations_vl,
                 segment_idx, title_prefix="EMG Detection Analysis"):
        
        self.time_gl = time_gl
        self.signal_gl = signal_gl
        self.energy_gl = energy_gl
        self.threshold_gl = threshold_gl
        self.activations_gl = activations_gl
        
        self.time_vl = time_vl
        self.signal_vl = signal_vl
        self.energy_vl = energy_vl
        self.threshold_vl = threshold_vl
        self.activations_vl = activations_vl
        
        self.segment_idx = segment_idx
        self.title_prefix = title_prefix
        
        # Create figure and initial layout
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f"{title_prefix} - Segment {segment_idx}", fontsize=16, fontweight='bold')
        
        # Create button area
        ax_button_gl = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_button_vl = plt.axes([0.25, 0.02, 0.1, 0.04])
        
        self.button_gl = Button(ax_button_gl, 'GL Analysis')
        self.button_vl = Button(ax_button_vl, 'VL Analysis')
        
        # Connect button events
        self.button_gl.on_clicked(self.show_gl)
        self.button_vl.on_clicked(self.show_vl)
        
        # Create subplot area (leave space for buttons)
        self.subplot_area = [0.1, 0.1, 0.85, 0.85]
        
        # Start with GL view
        self.current_view = 'GL'
        self.show_gl(None)
        
    def clear_subplots(self):
        """Clear existing subplots"""
        self.fig.clf()
        self.fig.suptitle(f"{self.title_prefix} - Segment {self.segment_idx}", fontsize=16, fontweight='bold')
        
        # Recreate buttons
        ax_button_gl = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_button_vl = plt.axes([0.25, 0.02, 0.1, 0.04])
        
        self.button_gl = Button(ax_button_gl, 'GL Analysis')
        self.button_vl = Button(ax_button_vl, 'VL Analysis')
        
        self.button_gl.on_clicked(self.show_gl)
        self.button_vl.on_clicked(self.show_vl)
        
        # Highlight current button
        if self.current_view == 'GL':
            self.button_gl.color = '0.85'
            self.button_vl.color = '0.95'
        else:
            self.button_gl.color = '0.95'
            self.button_vl.color = '0.85'
    
    def show_gl(self, event):
        """Show GL analysis"""
        self.current_view = 'GL'
        self.clear_subplots()
        
        # Create GL subplot
        axes = []
        axes.append(plt.axes([0.1, 0.7, 0.8, 0.2]))  # Signal
        axes.append(plt.axes([0.1, 0.45, 0.8, 0.2])) # Energy
        # axes.append(plt.axes([0.1, 0.2, 0.8, 0.2]))  # Crossings
        
        self._plot_detection_analysis(axes, self.time_gl, self.signal_gl, self.energy_gl, 
                                    self.threshold_gl, self.activations_gl, "GL")
        
        plt.draw()
    
    def show_vl(self, event):
        """Show VL analysis"""
        self.current_view = 'VL'
        self.clear_subplots()
        
        # Create VL subplot
        axes = []
        axes.append(plt.axes([0.1, 0.7, 0.8, 0.2]))  # Signal
        axes.append(plt.axes([0.1, 0.45, 0.8, 0.2])) # Energy
        # axes.append(plt.axes([0.1, 0.2, 0.8, 0.2]))  # Crossings
        
        self._plot_detection_analysis(axes, self.time_vl, self.signal_vl, self.energy_vl, 
                                    self.threshold_vl, self.activations_vl, "VL")
        
        plt.draw()
    
    def _plot_detection_analysis(self, axes, time, signal, energy_envelope, threshold, 
                               activations, muscle_name):
        """Plot the detection analysis for a specific muscle"""
        
        # Original signal
        axes[0].plot(time, signal, 'b-', alpha=0.7, label=f'{muscle_name} EMG Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{muscle_name} EMG Signal')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Energy envelope
        color = 'orange' if muscle_name == 'GL' else 'purple'
        axes[1].plot(time, energy_envelope, color, linewidth=2, label='Energy Envelope')
        axes[1].axhline(threshold, color='red', linestyle=':', linewidth=2, 
                       label=f'Threshold ({threshold:.2f})')
        
        # Mark detected activations
        if activations:
            for i, (on, off) in enumerate(activations):
                axes[1].axvspan(time[on], time[off], alpha=0.2, color='green', 
                              label=f'Activation {i+1}' if i == 0 else None)
                # Add vertical lines on signal plot
                axes[0].axvline(time[on], color='green', linestyle='--', linewidth=2,
                               label='Onset' if i == 0 else None)
                axes[0].axvline(time[off], color='red', linestyle='--', linewidth=2,
                               label='Offset' if i == 0 else None)
        
        axes[1].set_ylabel('Energy')
        axes[1].set_title('CWT Energy Envelope & Threshold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Threshold crossings
        # above_threshold = energy_envelope > threshold
        # axes[2].plot(time, above_threshold.astype(int), 'purple', linewidth=2, label='Above Threshold')
        # axes[2].fill_between(time, 0, above_threshold.astype(int), alpha=0.3, color='purple')
        
        # axes[2].set_ylabel('Binary')
        # axes[2].set_xlabel('Time (s)')
        # axes[2].set_title('Threshold Crossings')
        # axes[2].set_ylim(-0.1, 1.1)
        # axes[2].legend()
        # axes[2].grid(alpha=0.3)
        
        # Update legend on first plot if activations exist
        if activations:
            axes[0].legend()
    
    def show(self):
        """Display the plot"""
        plt.show()

def plot_tabbed_detection_debug(time_gl, signal_gl, energy_gl, threshold_gl, activations_gl,
                               time_vl, signal_vl, energy_vl, threshold_vl, activations_vl,
                               segment_idx, title_prefix="EMG Detection Analysis"):
    """
    Create a tabbed detection debug plot for both GL and VL.
    """
    tabbed_plot = TabbedDetectionPlot(time_gl, signal_gl, energy_gl, threshold_gl, activations_gl,
                                     time_vl, signal_vl, energy_vl, threshold_vl, activations_vl,
                                     segment_idx, title_prefix)
    tabbed_plot.show()

def plot_dwt(raw_sig, denoised_sig, fs, title="DWT Denoising"):
    t = np.arange(len(raw_sig)) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_sig, label="Raw", alpha=0.6)
    plt.plot(t, denoised_sig, label="Denoised", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_onset_offset(time, signal, onset, offset, title="Onset/Offset EMG"):
    plt.figure(figsize=(12,5))
    plt.plot(time, signal, label="EMG")
    plt.axvline(time[onset], color='green', linestyle='--', label="Onset")
    plt.axvline(time[offset], color='red', linestyle='--', label="Offset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_detection_debug(time, signal, energy_envelope, threshold, onset, offset, 
                        activations=None, title="Detection Debug"):
    """
    Debug plot showing the detection process step by step.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Original signal
    axes[0].plot(time, signal, 'b-', alpha=0.7, label='EMG Signal')
    if onset is not None and offset is not None:
        axes[0].axvline(time[onset], color='green', linestyle='--', linewidth=2, label='Onset')
        axes[0].axvline(time[offset], color='red', linestyle='--', linewidth=2, label='Offset')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original EMG Signal')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Energy envelope
    axes[1].plot(time, energy_envelope, 'orange', linewidth=2, label='Energy Envelope')
    axes[1].axhline(threshold, color='red', linestyle=':', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    # Mark detected activations
    if activations:
        for i, (on, off) in enumerate(activations):
            axes[1].axvspan(time[on], time[off], alpha=0.2, color='green', 
                          label=f'Activation {i+1}' if i == 0 else None)
    
    if onset is not None and offset is not None:
        axes[1].axvline(time[onset], color='green', linestyle='--', alpha=0.8)
        axes[1].axvline(time[offset], color='red', linestyle='--', alpha=0.8)
    
    axes[1].set_ylabel('Energy')
    axes[1].set_title('CWT Energy Envelope & Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Threshold crossings
    # above_threshold = energy_envelope > threshold
    # axes[2].plot(time, above_threshold.astype(int), 'purple', linewidth=2, label='Above Threshold')
    # axes[2].fill_between(time, 0, above_threshold.astype(int), alpha=0.3, color='purple')
    
    # if onset is not None and offset is not None:
    #     axes[2].axvline(time[onset], color='green', linestyle='--', alpha=0.8, label='Onset')
    #     axes[2].axvline(time[offset], color='red', linestyle='--', alpha=0.8, label='Offset')
    
    # axes[2].set_ylabel('Binary')
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_title('Threshold Crossings')
    # axes[2].set_ylim(-0.1, 1.1)
    # axes[2].legend()
    # axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Comprehensive final analysis plot with improved layout
def plot_comprehensive_analysis(full_time, filtered, cycles, segments_to_process, 
                               gait_segments, all_gl_activations, all_vl_activations):
    """
    Create the final comprehensive analysis plot with better organization.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Complete Gait Analysis - EMG Activation Detection", fontsize=18, fontweight='bold')
    
    # Create a more sophisticated layout
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Subplot 1: Gait Cycles
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(full_time, filtered["foot_switch"], 'r', linewidth=1.5, label='Foot Switch')
    ax1.set_title("Gait Cycles & Processed Segments", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Mark gait cycles (show more for better visualization)
    for i, (hs, to, hs_next) in enumerate(cycles[:20]):
        alpha_val = max(0.3, 1.0 - i*0.03)  # Fade out older cycles
        ax1.axvline(full_time[hs], color='green', linestyle='--', alpha=alpha_val, linewidth=1,
                   label='Heel Strike' if i == 0 else None)
        if to is not None:
            ax1.axvline(full_time[to], color='blue', linestyle='--', alpha=alpha_val, linewidth=1,
                       label='Toe Off' if i == 0 else None)
    
    # Mark processed segments
    for i, segment_idx in enumerate(segments_to_process):
        segment = gait_segments[segment_idx]
        start_time = full_time[segment['start_idx']]
        end_time = full_time[segment['end_idx']]
        ax1.axvspan(start_time, end_time, alpha=0.15, color='gray', 
                   label='Processed Segments' if i == 0 else None)
    
    ax1.legend(loc='upper right')
    
    # Subplot 2: GL Analysis
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(full_time, filtered["gl"], color="orange", linewidth=1, alpha=0.8, label='GL EMG')
    ax2.set_title("Gastrocnemius Lateralis - Activation Analysis", fontsize=14, fontweight='bold')
    ax2.set_ylabel("EMG (mV)", fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Mark GL activations with better visualization
    for i, (seg_idx, onset, offset, duration) in enumerate(all_gl_activations):
        onset_time = full_time[onset]
        offset_time = full_time[offset]
        
        # Use different colors for different segments        
        ax2.axvline(onset_time, color='green', linestyle='-', alpha=0.8, linewidth=2,
                   label='GL Onset' if i == 0 else None)
        ax2.axvline(offset_time, color='red', linestyle='-', alpha=0.8, linewidth=2,
                   label='GL Offset' if i == 0 else None)
        ax2.axvspan(onset_time, offset_time, alpha=0.2, color='green')
            
    ax2.legend(loc='upper right')
    
    # Subplot 3: VL Analysis
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(full_time, filtered["vl"], color="brown", linewidth=1, alpha=0.8, label='VL EMG')
    ax3.set_title("Vastus Lateralis - Activation Analysis", fontsize=14, fontweight='bold')
    ax3.set_ylabel("EMG (mV)", fontsize=12)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.grid(alpha=0.3)
    
    # Mark VL activations
    for i, (seg_idx, onset, offset, duration) in enumerate(all_vl_activations):
        onset_time = full_time[onset]
        offset_time = full_time[offset]
        
        ax3.axvline(onset_time, color='green', linestyle='-', alpha=0.8, linewidth=2,
                   label='VL Onset' if i == 0 else None)
        ax3.axvline(offset_time, color='red', linestyle='-', alpha=0.8, linewidth=2,
                   label='VL Offset' if i == 0 else None)
        ax3.axvspan(onset_time, offset_time, alpha=0.2, color='blue')
        
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
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