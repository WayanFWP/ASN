import matplotlib.pyplot as plt
import numpy as np
from preproceed import feature_data

def plotSignal(signal, fs):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    time = np.arange(signal.shape[2]) / fs

    for i, ax in enumerate(axes):
        ax.plot(time, signal[0, i, :])
        ax.set_title(f'Channel: {feature_data[i]}')
        ax.set_ylabel('Amplitude (µV)')

    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    # plt.show()
    return fig

def plot2Signal(signal1, signal2, fs, label1, label2, idx=0, unit="power"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex=True)

    time = np.arange(signal1.shape[2]) / fs
    time = time - 4.0  # cue at t=0

    for i in range(3):
        axes[i, 0].plot(time, signal1[idx, i, :], color='b')
        axes[i, 0].axhline(0, linestyle='--', linewidth=1)
        axes[i, 0].set_title(f'{label1} - {feature_data[i]}')
        if unit == 'power':
            axes[i, 0].set_ylabel('Power change (%)')
        else:
            axes[i, 0].set_ylabel('Amplitude (µV)')

        axes[i, 1].plot(time, signal2[idx, i, :], color='r')
        axes[i, 1].axhline(0, linestyle='--', linewidth=1)
        axes[i, 1].set_title(f'{label2} - {feature_data[i]}')
        if unit == 'power':
            axes[i, 1].set_ylabel('Power change (%)')
        else:
            axes[i, 1].set_ylabel('Amplitude (µV)')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig

def merge2Signals(signal1, signal2, fs, label1='Signal 1', label2='Signal 2', unit='power', idx = 0):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    time = np.arange(signal1.shape[2]) / fs
    # X axis is -4 to 3 seconds
    time = time - 4.0  # since tmin=-4 and tmax=3
    for i in range(3):
        axes[i].plot(time, signal1[idx, i, :], color='b', label=label1)
        axes[i].plot(time, signal2[idx, i, :], color='r', label=label2)
        axes[i].axhline(0, linestyle='--', linewidth=1)
        axes[i].set_title(f'Channel: {feature_data[i]}')
        if unit == 'power':
            axes[i].set_ylabel('Power change (%)')
        else:
            axes[i].set_ylabel('Amplitude (µV)')
        axes[i].legend()
    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    # plt.show()
    return fig


def plot_topographic_map(data, fs, time_points, title="ERD/ERS Topography"):
    """
    Plot topographic maps at specific time points.
    
    Parameters:
    -----------
    data : ndarray (n_trials, n_channels, n_times)
    fs : int, sampling frequency
    time_points : list of time points in seconds (e.g., [0, 1, 2])
    """
    import matplotlib.patches as patches
    
    # Average across trials
    avg_data = np.mean(data, axis=0)  # (3, n_times)
    
    # Channel positions (C3, Cz, C4 in 2D)
    positions = {
        "EEG:C3": (-0.5, 0),
        "EEG:Cz": (0, 0),
        "EEG:C4": (0.5, 0)
    }
    
    n_time_points = len(time_points)
    fig, axes = plt.subplots(1, n_time_points, figsize=(4*n_time_points, 3))
    
    if n_time_points == 1:
        axes = [axes]
    
    for idx, t_sec in enumerate(time_points):
        t_idx = int((t_sec + 4.0) * fs)  # convert to sample index
        
        ax = axes[idx]
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        
        # Plot channels as colored circles
        for ch_idx, ch_name in enumerate(feature_data):
            x, y = positions[ch_name]
            value = avg_data[ch_idx, t_idx]
            
            # Color based on ERD (negative) or ERS (positive)
            color = 'blue' if value < 0 else 'red'
            size = abs(value) * 20  # scale size
            
            circle = patches.Circle((x, y), radius=0.15, 
                                   color=color, alpha=0.6)
            ax.add_patch(circle)
            ax.text(x, y, f'{value:.1f}%', 
                   ha='center', va='center', fontsize=10)
        
        ax.set_title(f't = {t_sec}s')
        ax.axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig