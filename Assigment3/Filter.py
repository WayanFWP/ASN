import numpy as np
from utils import *
import pywt

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

    def cwt(self, signal, scales):
        F = 0.849
        W0 = 2 * np.pi * F
        
        n = len(signal)
        n_scales = len(scales)
        cwt_matrix = np.zeros((n_scales, n), dtype=np.complex128)

        for i, s in enumerate(scales):
            wavelet_len = int(10 * s * self.fs)
            if wavelet_len % 2 == 0: 
                wavelet_len += 1
            t = (np.arange(wavelet_len) - wavelet_len//2) / self.fs

            # scaled wavelet ψ((t)/s)
            psi_s = self.morlet_wavelet(t / s, W0)

            # convolution implementing ∫ x(t) ψ*((t-τ)/s) dt
            conv = np.convolve(signal, np.conj(psi_s[::-1]), mode="same")

            # apply scale normalization 1/sqrt(s)
            cwt_matrix[i, :] = conv / np.sqrt(s)

        frequencies = F / scales
        return cwt_matrix, frequencies

    def cwt_analysis(self, pcg_segment):
        if len(pcg_segment) == 0: 
            return None, None, None
        
        segment_duration = len(pcg_segment) / self.fs
        MAX_SCALE_SEC = min(0.1, segment_duration / 20)  # Limit max scale
        MIN_SCALE_SEC = 0.002
        
        total_scales = 128
        scales_a = np.logspace(np.log10(MIN_SCALE_SEC), np.log10(MAX_SCALE_SEC), num=total_scales)
        coefficients, frequencies = self.cwt(pcg_segment, scales_a)
        
        n_samples = len(pcg_segment)
        center_sample = n_samples // 2
        time_axis_segment = (np.arange(n_samples) - center_sample) / self.fs
        
        return coefficients, frequencies, time_axis_segment
    
    def dwt_denoise(self, signal):
        wavelet = 'db4'
        level = 8

        coeffs = pywt.wavedec(signal, wavelet, level=level)

        threshold = np.sqrt(2*np.log(len(signal)))
        
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c 
                            for i, c in enumerate(coeffs)]
        
        denoised = pywt.waverec(coeffs_thresholded, wavelet)
        denoised = denoised[:len(signal)]
        
        return denoised
