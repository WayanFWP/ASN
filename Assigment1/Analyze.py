import numpy as np
from scipy.signal import find_peaks

class HeartRate:
    def __init__(self, fs):
        self.fs = fs

    def analyze(self, signal_data):
        # Process the heart rate signal data
        min_dist_sample = int(self.fs * 0.4)  # Minimum distance between peaks in samples
        peaks, _ = find_peaks(signal_data, height=0, distance=max(1, min_dist_sample), prominence=1.2 * np.std(signal_data))

        peaktime_sec = peaks / self.fs
        rr_intervals = np.diff(peaktime_sec)
        bpm = 60 / np.mean(rr_intervals)  # Beats per minute
        
        return signal_data, peaks, bpm

class Respiratory:
    def __init__(self, fs):
        self.fs = fs
        self.signal_data = None

    def analyze(self, signal_data):
        self.signal_data = signal_data
        
        # Process the respiratory signal data
        min_dist_sample = int(self.fs * 0.5)  # Minimum distance between peaks in samples
        # peaks, _ = find_peaks(signal_data, height=0, distance=max(1, min_dist_sample), prominence=1.25 * np.std(signal_data))
        peaks, _ = find_peaks(signal_data, height=0, distance=max(1, min_dist_sample), prominence=1. * np.std(signal_data))
        # peaks, _ = find_peaks(signal_data, height=0, distance=max(1, min_dist_sample), prominence=.25 * np.std(signal_data))
        
        duration_s = len(signal_data) / self.fs
        num_peaks = len(peaks)
        bpm = (num_peaks / duration_s) * 60  # Breaths per minute
        
        return signal_data, peaks, bpm

    def get_freq(self):
        N = len(self.signal_data)
        freq = np.fft.rfftfreq(N, d=1./self.fs)
        fft_values = np.fft.rfft(self.signal_data)
        magnitude = np.abs(fft_values) / N
        
        peak_index = np.argmax(magnitude[1:]) + 1  # Ignore DC component
        peak_freq = freq[peak_index]
        peak_mag = magnitude[peak_index]
        return freq, magnitude, peak_freq, peak_mag
        
class Vasometric:
    def __init__(self, fs):
        self.fs = fs

    def analyze(self, signal_data):
        # Analyze vasomotor signal data
        N = len(signal_data)
        freq = np.fft.rfftfreq(N, d=1./self.fs)
        fft_values = np.fft.rfft(signal_data)
        magnitude = np.abs(fft_values) / N

        peak_index = np.argmax(magnitude[1:]) + 1  # Ignore DC component
        peak_freq = freq[peak_index]
        peak_mag = magnitude[peak_index]

        return freq, magnitude, peak_freq, peak_mag