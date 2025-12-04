import numpy as np
from utils import *

class Filter:
    def __init__(self, fs):
        self.fs = fs
        
    def LPF(self, signal, cutoff):
        N = len(signal)
        T = 1 / self.fs
        Wc = 2 * np.pi * cutoff

        denom = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
        b1 = ((8 / T**2) - (2 * Wc**2)) / denom
        b2 = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denom
        a0 = Wc**2 / denom
        a1 = 2 * Wc**2 / denom
        a2 = a0
        y = np.zeros(N)
        for n in range(2, N-2):
            y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
        return y
    
    def HPF(self, signal, cutoff):
        N = len(signal)
        T = 1/self.fs
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

    def BPF(self, signal, lowcut, highcut):
        lowpass = self.LPF(signal, highcut)
        bandpass = self.HPF(lowpass, lowcut)
        return bandpass

    def morlet_wavelet(self, t, W0):
        norm_factor = np.pi ** (-0.25)
        sinusoid = np.exp(1j * W0 * t)
        gaussian = np.exp(-0.5 * t**2)
        return norm_factor * sinusoid * gaussian

    def cwt(self, signal, scales, percentage_keep=None):
        F = 0.849
        W0 = 2 * np.pi * F

        n = len(signal)
        n_scales = len(scales)
        cwt_matrix = np.zeros((n_scales, n), dtype=np.complex128)

        for i, s in enumerate(scales):
            wavelet_len = min(int(10 * s * self.fs), n * 2)
            if wavelet_len % 2 == 0:
                wavelet_len += 1

            t = (np.arange(wavelet_len) - wavelet_len // 2) / self.fs
            psi_s = self.morlet_wavelet(t / s, W0)

            conv = np.convolve(signal, np.conj(psi_s[::-1]), mode="same")

            if len(conv) != n:
                if len(conv) > n:
                    start_idx = (len(conv) - n) // 2
                    conv = conv[start_idx:start_idx + n]
                else:
                    pad_width = n - len(conv)
                    conv = np.pad(conv, (pad_width // 2, pad_width - pad_width // 2))

            conv = conv / np.sqrt(s)

            if percentage_keep is not None:
                p = 100 * (1 - percentage_keep)
                thr = np.percentile(np.abs(conv), p)
                conv[np.abs(conv) < thr] = 0

            cwt_matrix[i, :] = conv

        frequencies = F / scales
        return cwt_matrix, frequencies

    def cwt_analysis(self, segmented_signals, percentage_keep=99):
        if len(segmented_signals) == 0: 
            return None, None, None
        
        segment_duration = len(segmented_signals) / self.fs
        MAX_SCALE_SEC = min(0.1, segment_duration / 20) 
        MIN_SCALE_SEC = 0.002
        
        total_scales = 128
        scales_a = np.logspace(np.log10(MIN_SCALE_SEC), np.log10(MAX_SCALE_SEC), num=total_scales)
        coefficients, frequencies = self.cwt(segmented_signals, scales_a, percentage_keep=percentage_keep)
        
        n_samples = len(segmented_signals)
        center_sample = n_samples // 2
        time_axis_segment = (np.arange(n_samples) - center_sample) / self.fs
        
        return coefficients, frequencies, time_axis_segment
        