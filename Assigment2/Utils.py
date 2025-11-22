from scipy.signal import butter, lfilter, find_peaks
import numpy as np
import scipy.ndimage as ndi

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    low_passed = LPF(data, highcut, fs)
    band_passed = HPF(low_passed, lowcut, fs)
    return band_passed

def LPF(signal, fl, fs):
    N = len(signal)
    T = 1 / fs
    Wc = 2 * np.pi * fl

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

def HPF(signal,fh,fs):
    N = len(signal)
    T = 1/fs
    Wc = 2 * np.pi * fh

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

def pan_tompkins(ecg, fs):
    # 1) Bandpass filter between 5-15 Hz
    filtered = bandpass_filter(ecg, 5, 15, fs, order=3)
    T = 1 / fs
    
    # 2) Derivative filter (5-point derivative)
    # y(n) = (1/8T)[x(n+2) + 2x(n+1) - 2x(n-1) - x(n-2)]
    h = np.array([1, 2, 0, -2, -1]) * (1/(T * 8.0))
    derivative = np.convolve(filtered, h, mode='same')
    
    # 3) Squaring
    squared = derivative ** 2
    
    # 4) Moving window integration
    # Window size: ~150ms width
    win_size = int(0.150 * fs)
    window = np.ones(win_size) / win_size
    integrated = np.convolve(squared, window)
    
    # 5) Find peaks in integrated signal
    min_distance = int(0.2 * fs) 
    
    # Use adaptive threshold
    threshold = 0.012 * np.max(integrated)
    peaks, properties = find_peaks(integrated, 
                                   distance=min_distance,
                                   height=threshold)
    
    # 6) Refine peaks by finding max in original filtered signal
    r_peaks = []
    search_window = int(0.075 * fs)  # ±75ms search window
    
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(filtered), peak + search_window)
        local_segment = np.abs(filtered[start:end])
        local_max = start + np.argmax(local_segment)
        r_peaks.append(local_max)
    
    r_peaks = np.array(r_peaks)
    
    # Remove duplicates
    # r_peaks = np.unique(r_peaks)
    
    return filtered, r_peaks

def segment(signal, r_peaks):
    segments = []
    for i in range(len(r_peaks) - 1):
        start_idx = r_peaks[i]
        end_idx = r_peaks[i + 1]
        segment = signal[start_idx:end_idx]
        
        segments.append({
            'segment': segment,
            'start_r_peak': r_peaks[i],
            'end_r_peak': r_peaks[i + 1],
            'duration_samples': end_idx - start_idx,
            'cycle_index': i
        })
    
    return segments

def morlet_wavelet_delphi(t, W0):
    norm_factor = np.pi ** (-0.25)
    sinusoid = np.exp(1j * W0 * t)
    gaussian = np.exp(-0.5 * t**2)
    return norm_factor * sinusoid * gaussian

def cwt(signal, fs, scales):
    F = 0.849
    W0 = 2 * np.pi * F
    
    n = len(signal)
    n_scales = len(scales)
    cwt_matrix = np.zeros((n_scales, n), dtype=np.complex128)

    for i, s in enumerate(scales):
        wavelet_len = int(10 * s * fs)
        if wavelet_len % 2 == 0: 
            wavelet_len += 1
        t = (np.arange(wavelet_len) - wavelet_len//2) / fs

        # scaled wavelet ψ((t)/s)
        psi_s = morlet_wavelet_delphi(t / s, W0)

        # convolution implementing ∫ x(t) ψ*((t-τ)/s) dt
        conv = np.convolve(signal, np.conj(psi_s[::-1]), mode="same")

        # apply scale normalization 1/sqrt(s)
        cwt_matrix[i, :] = conv / np.sqrt(s)

    frequencies = F / scales
    return cwt_matrix, frequencies

def cwt_analysis(pcg_segment, fs):
    if len(pcg_segment) == 0: return None, None, None
    MIN_SCALE_SEC, MAX_SCALE_SEC = 0.002, 0.1
    total_scales = 128
    scales_a = np.logspace(np.log10(MIN_SCALE_SEC), np.log10(MAX_SCALE_SEC), num=total_scales)
    coefficients, frequencies = cwt(pcg_segment, fs, scales_a)
    time_axis_segment = np.arange(len(pcg_segment)) / fs
    return coefficients, frequencies, time_axis_segment


def stft_analysis(pcg_segment, fs, window_length_ms, overlap_pct):
    if len(pcg_segment) == 0: return None, None, None
    n_per_seg = int(window_length_ms / 1000 * fs)
    n_overlap = int(n_per_seg * (overlap_pct / 100))
    hop_length = n_per_seg - n_overlap

    # window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_per_seg) / (n_per_seg - 1))  # Hanning window
    # window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_per_seg) / (n_per_seg - 1))  # Hamming window
    window = 1 - abs(2 * np.arange(n_per_seg) - (n_per_seg + 1)) / (n_per_seg - 1)  # Triangular window
    
    stft_frames = []
    num_frames = 1 + int(np.floor((len(pcg_segment) - n_per_seg) / hop_length))
    n_fft = 0
    for i in range(num_frames):
        start_index = i * hop_length
        end_index = start_index + n_per_seg
        frame = pcg_segment[start_index:end_index]
        if len(frame) < n_per_seg:
            frame = np.pad(frame, (0, n_per_seg - len(frame)), 'constant', constant_values=0)
        windowed_frame = frame * window
        fft_result = np.fft.fft(windowed_frame)
        if i == 0: n_fft = len(fft_result)
        magnitude = np.abs(fft_result)[:n_fft // 2]
        stft_frames.append(magnitude)
    if not stft_frames: return None, None, None
    spectrogram = np.array(stft_frames).T
    freqs = np.fft.fftfreq(n_fft, d=1/fs)[:n_fft // 2]
    times = np.arange(num_frames) * hop_length / fs
    return freqs, times, spectrogram

def compute_cog(mask_blob, times, scales, E):
        Em = E * mask_blob
        E_t = np.sum(Em, axis=0)
        E_s = np.sum(Em, axis=1)
        if np.sum(E_t) == 0 or np.sum(E_s) == 0:
            return None, None
        t_cog = np.sum(times * E_t) / np.sum(E_t)
        s_cog = np.sum(scales * E_s) / np.sum(E_s)
        return t_cog, s_cog
    
def flood_fill(si, ti, label, mask, labels, S, T):
    stack = [(si, ti)]
    while stack:
        i, j = stack.pop()
        if i < 0 or i >= S or j < 0 or j >= T:
            continue
        if (not mask[i, j]) or labels[i, j] != 0:
            continue
        labels[i, j] = label
        stack.append((i-1, j))
        stack.append((i+1, j))
        stack.append((i, j-1))
        stack.append((i, j+1))


def detect_s1_s2(coeff, scales, times, f0_delphi=0.849):
    E = np.abs(coeff)**2
    S, T = E.shape

    # ======================================================
    # STEP 1 : PRE-SPLIT WAKTU S1 & S2 BERDASARKAN ENVELOPE
    # ======================================================
    energy_time = np.sum(E, axis=0)
    t_peak1 = times[np.argmax(energy_time)]
    time_boundary = t_peak1 + 0.2     # 150 ms after S1 (adjustable)

    # ======================================================
    # STEP 2 : THRESHOLDING UNTUK S1 & S2
    # ======================================================
    thr_s1 = 0.60 * np.max(E)
    thr_s2 = 0.15 * np.max(E)

    mask_s1 = (E > thr_s1) & (times[None, :] < time_boundary)
    mask_s2 = (E > thr_s2) & (times[None, :] >= time_boundary)

    mask = mask_s1 | mask_s2

    # ======================================================
    # STEP 3 : CONNECTED COMPONENT LABELING
    # ======================================================
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0

    for i in range(S):
        for j in range(T):
            if mask[i, j] and labels[i, j] == 0:
                current_label += 1
                flood_fill(i, j, current_label, mask, labels, S, T)

    if current_label < 2:
        print("Warning: fewer than 2 heart sound regions.")
        return None

    # ======================================================
    # STEP 4 : SORT REGION BERDASARKAN WAKTU CoG
    # ======================================================
    regions = []
    for lab in range(1, current_label + 1):
        m = (labels == lab)
        Em = E * m
        E_t = np.sum(Em, axis=0)
        if np.sum(E_t) == 0:
            continue
        t_c = np.sum(times * E_t) / np.sum(E_t)
        regions.append((t_c, lab))

    regions.sort(key=lambda x: x[0])
    label_s1 = regions[0][1]
    S1_mask = (labels == label_s1)
    # ======================================================
    # STEP 5 : HITUNG COG
    # ======================================================
    
    t_s1, s_s1 = compute_cog(S1_mask, times, scales, E)
    f_s1 = f0_delphi / s_s1

    label_s2 = regions[1][1]
    S2_mask = (labels == label_s2)
    t_s2, s_s2 = compute_cog(S2_mask, times, scales, E)
    f_s2 = f0_delphi / s_s2
    return (t_s1, f_s1, S1_mask), (t_s2, f_s2, S2_mask), E
