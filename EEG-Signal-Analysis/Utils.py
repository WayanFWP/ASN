import numpy as np

def norm(x):
    min_val = np.min(x)
    max_val = np.max(x)
    norm_result = (x - min_val) / (max_val - min_val)
    return norm_result

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

def morlet_wavelet(t, W0):
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
        psi_s = morlet_wavelet(t / s, W0)

        # convolution implementing ∫ x(t) ψ*((t-τ)/s) dt
        conv = np.convolve(signal, np.conj(psi_s[::-1]), mode="same")

        # apply scale normalization 1/sqrt(s)
        cwt_matrix[i, :] = conv / np.sqrt(s)

    frequencies = F / scales
    return cwt_matrix, frequencies

def cwt_analysis(pcg_segment, fs):
    # Handle both 1D and 2D input
    if pcg_segment.ndim == 1:
        # Single channel case
        return _cwt_single_channel(pcg_segment, fs)
    elif pcg_segment.ndim == 2:
        # Multiple channels
        n_channels = pcg_segment.shape[0]
        cwt_results = []
        
        for ch in range(n_channels):
            cwt_ch, freq_ch, time_ch = _cwt_single_channel(pcg_segment[ch, :], fs)
            cwt_results.append(cwt_ch)
        
        # Stack all channels: shape (n_channels, n_scales, n_times)
        coefficients = np.array(cwt_results)
        frequencies = freq_ch  # Same for all channels
        time_axis = time_ch     # Same for all channels
        
        return coefficients, frequencies, time_axis
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {pcg_segment.shape}")

def _cwt_single_channel(pcg_segment, fs):
    """Helper function for single channel CWT"""
    if len(pcg_segment) == 0: 
        return None, None, None
    
    # Adjust scale range based on signal length
    segment_duration = len(pcg_segment) / fs
    MAX_SCALE_SEC = min(0.1, segment_duration / 20)
    MIN_SCALE_SEC = 0.002
    
    total_scales = 128
    scales_a = np.logspace(np.log10(MIN_SCALE_SEC), np.log10(MAX_SCALE_SEC), num=total_scales)
    coefficients, frequencies = cwt(pcg_segment, fs, scales_a)
    
    # Create time axis relative to center
    n_samples = len(pcg_segment)
    center_sample = n_samples // 2
    time_axis_segment = (np.arange(n_samples) - center_sample) / fs
    
    return coefficients, frequencies, time_axis_segment