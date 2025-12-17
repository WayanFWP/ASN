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
    plt.show()

def plot2Signal(signal1, signal2, fs, label1='Signal 1', label2='Signal 2'):
    fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex=True)
    time = np.arange(signal1.shape[2]) / fs

    for i in range(3):
        axes[i, 0].plot(time, signal1[0, i, :], color='b')
        axes[i, 0].set_title(f'{label1} - Channel: {feature_data[i]}')
        axes[i, 0].set_ylabel('Amplitude (µV)')

        axes[i, 1].plot(time, signal2[0, i, :], color='r')
        axes[i, 1].set_title(f'{label2} - Channel: {feature_data[i]}')
        axes[i, 1].set_ylabel('Amplitude (µV)')
    axes[-1, 0].set_xlabel('Time (samples)')
    axes[-1, 1].set_xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()