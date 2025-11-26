from scipy.signal import find_peaks
import numpy as np

def bandpass_filter(data, lowcut, highcut, fs):
    low_passed = HPF(data, lowcut, fs)
    band_passed = LPF(low_passed, highcut, fs)
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
    filtered = bandpass_filter(ecg, 5, 15, fs)
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
    threshold = 0.05 * np.max(integrated)
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
    
    return filtered, r_peaks

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
    if len(pcg_segment) == 0: 
        return None, None, None
    
    # Adjust scale range based on signal length
    segment_duration = len(pcg_segment) / fs
    MAX_SCALE_SEC = min(0.1, segment_duration / 20)  # Limit max scale
    MIN_SCALE_SEC = 0.002
    
    total_scales = 128
    scales_a = np.logspace(np.log10(MIN_SCALE_SEC), np.log10(MAX_SCALE_SEC), num=total_scales)
    coefficients, frequencies = cwt(pcg_segment, fs, scales_a)
    
    # Create time axis relative to center (R-peak position)
    # Assuming R-peak is at the center of the segment
    n_samples = len(pcg_segment)
    center_sample = n_samples // 2
    time_axis_segment = (np.arange(n_samples) - center_sample) / fs
    
    return coefficients, frequencies, time_axis_segment

def windowing(method, n_per_seg):
    if method == "hanning":
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_per_seg) / (n_per_seg - 1))  # Hanning window
    elif method == "hamming":
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_per_seg) / (n_per_seg - 1))  # Hamming window
    elif method == "triangular":
        window = 1 - abs(2 * np.arange(n_per_seg) - (n_per_seg + 1)) / (n_per_seg - 1)  # Triangular window
    else:
        window = np.ones(n_per_seg)  # Rectangular window
    return window

def stft(x, window_size, hop_size, window='hann'):
    # Input validation
    if len(x) < window_size:
        raise ValueError("Signal length must be at least window size")
    
    if hop_size <= 0 or hop_size > window_size:
        raise ValueError("Hop size must be between 1 and window size")
    
    window_func = windowing(method=window, n_per_seg=window_size)
    
    # Normalize window to preserve energy
    window_func = window_func / np.sqrt(np.sum(window_func**2))
    
    # Calculate number of frames
    num_frames = 1 + (len(x) - window_size) // hop_size
    
    # Initialize STFT matrix
    stft_matrix = np.zeros((window_size, num_frames), dtype=complex)
    
    # Compute STFT for each frame
    for frame_idx in range(num_frames):
        # Extract segment
        start_idx = frame_idx * hop_size
        end_idx = start_idx + window_size
        segment = x[start_idx:end_idx]
        
        # Apply window
        windowed_segment = segment * window_func
        
        # Compute FFT
        stft_matrix[:, frame_idx] = np.fft.fft(windowed_segment)
    
    # Calculate frequency and time arrays
    frequencies = np.fft.fftfreq(window_size)
    time_frames = np.arange(num_frames) * hop_size
    
    return stft_matrix, frequencies, time_frames
    

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

def detect_s1_s2(coeff, scales, times, f0_delphi=0.849, thresh_s1=0.6, thresh_s2=0.15):
    E = np.abs(coeff)**2
    S, T = E.shape
        
    s1_start = -0.1  
    s1_end = 0.2        
    s2_start = 0.2        
    s2_end = 0.4        
    
    thr_s1 = thresh_s1 * np.max(E)
    thr_s2 = thresh_s2 * np.max(E)

    mask_s1 = (E > thr_s1) & (times[None, :] >= s1_start) & (times[None, :] < s1_end)
    mask_s2 = (E > thr_s2) & (times[None, :] >= s2_start) & (times[None, :] < s2_end)

    mask = mask_s1 | mask_s2
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0

    for i in range(S):
        for j in range(T):
            if mask[i, j] and labels[i, j] == 0:
                current_label += 1
                flood_fill(i, j, current_label, mask, labels, S, T)

    if current_label < 1:
        print("Warning: No heart sound regions detected.")
        return None
    
    regions = []
    for lab in range(1, current_label + 1):
        m = (labels == lab)
        Em = E * m
        E_t = np.sum(Em, axis=0)
        if np.sum(E_t) == 0:
            continue
        t_c = np.sum(times * E_t) / np.sum(E_t)
        
        if s1_start <= t_c < s1_end:
            region_type = 'S1'
        elif s2_start <= t_c < s2_end:
            region_type = 'S2'
        else:
            region_type = 'Unknown'
            
        regions.append((t_c, lab, region_type))

    regions.sort(key=lambda x: x[0])
    
    s1_region = None
    s2_region = None
    
    for t_c, lab, region_type in regions:
        if region_type == 'S1' and s1_region is None:
            s1_region = (t_c, lab)
        elif region_type == 'S2' and s2_region is None:
            s2_region = (t_c, lab)
    
    if s1_region is None:
        print("Warning: No S1 region detected in expected time window")
        return None
    
    # Process S1
    label_s1 = s1_region[1]
    S1_mask = (labels == label_s1)
    
    # Use 1/scales for consistent coordinate system with plotting
    scales_plot = 1/scales  # Convert to plotting coordinates
    t_s1, scale_plot_s1 = compute_cog(S1_mask, times, scales_plot, E)
    
    # Calculate actual frequency from the plotting scale
    f_s1 = 1 / scale_plot_s1

    if s2_region is not None:
        label_s2 = s2_region[1]
        S2_mask = (labels == label_s2)
        t_s2, scale_plot_s2 = compute_cog(S2_mask, times, scales_plot, E)
        f_s2 = 1 / scale_plot_s2
        
        print(f"Detected both S1 and S2")
        # Return plotting scale coordinates directly
        return (t_s1, scale_plot_s1, S1_mask), (t_s2, scale_plot_s2, S2_mask), E
    else:
        print(f"Only S1 detected, no S2 found in expected time window")
        return (t_s1, scale_plot_s1, S1_mask), None, E
