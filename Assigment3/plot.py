# plot.py
import numpy as np
import matplotlib.pyplot as plt

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
