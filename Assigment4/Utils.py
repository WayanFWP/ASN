import numpy as np
import scipy.signal as signal

def LPF(signal, cutoff, fs):
    N = len(signal)
    T = 1 / fs
    Wc = 2 * np.pi * cutoff

    denom = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
    b1 = ((8 / T**2) - (2 * Wc**2)) / denom
    b2 = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denom
    a0 = Wc**2 / denom
    a1 = 2 * Wc**2 / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(0, N-1):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y

def HPF(signal, cutoff, fs):
    N = len(signal)
    T = 1/fs
    Wc = 2 * np.pi * cutoff

    denom = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/ denom
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/ denom
    a0 = (4/T**2) / denom
    a1 = (-8/T**2) / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(0, N-1):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y

def BPF(signal, lowcut=0.5, highcut=30, fs=250):
    x = HPF(signal, lowcut, fs)
    bandpassed = LPF(x, highcut, fs)
    return bandpassed

def squaring(signal):
    return signal ** 2

def avaragingOverN(signal, N=10):
    N = int(N)
    kernel = np.ones(N) / N
    padded_signal = np.pad(signal, (N-1, 0), mode='edge')
    averaged_signal = np.convolve(padded_signal, kernel, mode='valid')
    return averaged_signal