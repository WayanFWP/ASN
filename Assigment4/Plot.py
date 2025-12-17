import matplotlib.pyplot as plt
import numpy as np
from preproceed import feature_data

def plot_signal(signal, fs):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    time = np.arange(signal.shape[2]) / fs

    for i, ax in enumerate(axes):
        ax.plot(time, signal[0, i, :])
        ax.set_title(f'Channel: {feature_data[i]}')
        ax.set_ylabel('Amplitude (µV)')

    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()
