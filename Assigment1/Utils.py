import numpy as np
import ctypes
from scipy.signal import welch
import matplotlib.pyplot as plt

class Complex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

# Load .so
convo = ctypes.CDLL("/home/nayaw/project/ASN/Assigment1/Acceleration/convolution.so")
fft = ctypes.CDLL("/home/nayaw/project/ASN/Assigment1/Acceleration/FFT.so")

# C function signature: double* convolution(const double*, int, const double*, int)
convo.convolution.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]
convo.convolution.restype = ctypes.POINTER(ctypes.c_double)  # returns double*

# C function signature: double* fft(const double* x, int N)
fft.fft.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
                    ctypes.c_int]
fft.fft.restype = ctypes.POINTER(Complex128)
fft.free_memory.argtypes = [ctypes.c_void_p]
fft.free_memory.restype = None

def dirac(x):
    return 1 if x == 0 else 0

def downSample(x, M):
    if M <= 0:
        raise ValueError("Downsampling factor M must be positive.")
    
    # y[n] = x[nM]
    y = np.array([x[n * M] for n in range(len(x) // M)])
    return y

def convolve(x, h):
    signal = np.ascontiguousarray(x, dtype=np.float64)
    filt = np.ascontiguousarray(h, dtype=np.float64)
    signal_len, filt_len = len(signal), len(filt)
    conv_len = signal_len + filt_len - 1
    
    result_ptr = convo.convolution(
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), signal_len,
        filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), filt_len
    )
    
    # Convert result back to numpy array
    result = np.array([result_ptr[i] for i in range(conv_len)])
    
    # Free the allocated memory
    convo.free(result_ptr)
    return result

def count_zero_crossings(sig):
    crossings = np.where(np.diff(np.sign(sig)))[0]
    return len(crossings) // 2  # each full cycle = 2 crossings

def FFT(sig, fs):
    # Convert real signal to complex array
    N = len(sig)
    x = np.ascontiguousarray(sig, dtype=np.complex128)
    
    result_ptr = fft.fft(x, N)
    
    # Convert result back to numpy array
    raw = np.ctypeslib.as_array(result_ptr, shape=(N,))
    result = raw['real'] + 1j * raw['imag']
    fft.free_memory(result_ptr)
        
    magnitude = np.abs(result[:N//2]) / (N/2)
    freq = np.arange(N//2) * fs / N
    return magnitude, freq

def hrv_time_domain(rr_intervals, rr_times=None):
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    if rr_intervals.size < 2:
        raise ValueError("Butuh minimal 2 RR interval untuk menghitung HRV.")

    N = rr_intervals.size

    if rr_times is None:
        rr_times = np.concatenate(([0.0], np.cumsum(rr_intervals)))[1:]
    else:
        rr_times = np.asarray(rr_times, dtype=float)
        if rr_times.size == rr_intervals.size + 1:
            rr_times = rr_times[1:]

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    mean_hr = 60.0 / mean_rr

    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    sdsd = np.std(diff_rr, ddof=1)

    thr = 0.05
    nn50 = np.sum(np.abs(diff_rr) > thr)
    pnn50 = 100.0 * nn50 / len(diff_rr) if len(diff_rr) > 0 else np.nan

    cvnn = (np.std(rr_intervals, ddof=0) / mean_rr) * 100.0
    cvsd = (np.std(diff_rr, ddof=0) / mean_rr) * 100.0 if len(diff_rr)>0 else np.nan

    skewness = np.mean(((rr_intervals - mean_rr) / np.std(rr_intervals))**3) if np.std(rr_intervals)>0 else 0.0

    # SDANN dan SDNN index (5 menit)
    seg_length_seconds = 300.0
    total_duration = rr_times[-1] - rr_times[0]
    sdann = np.nan
    sdnn_index = np.nan
    if total_duration >= seg_length_seconds:
        t0 = rr_times[0]
        segments_means = []
        segments_stds = []
        seg_start = t0
        while seg_start + seg_length_seconds <= rr_times[-1]:
            seg_end = seg_start + seg_length_seconds
            mask = (rr_times >= seg_start) & (rr_times < seg_end)
            seg_rr = rr_intervals[mask]
            if seg_rr.size > 0:
                segments_means.append(np.mean(seg_rr))
                segments_stds.append(np.std(seg_rr, ddof=1) if seg_rr.size>1 else 0.0)
            seg_start = seg_end
        if len(segments_means) >= 1:
            sdann = np.std(np.asarray(segments_means), ddof=0)
        if len(segments_stds) >= 1:
            sdnn_index = np.mean(np.asarray(segments_stds))

    bin_width = 0.0078125
    counts, bin_edges = np.histogram(rr_intervals, bins=int((np.max(rr_intervals)-np.min(rr_intervals))/bin_width))
    max_count = counts.max() if counts.size>0 else np.nan
    hti = len(rr_intervals) / max_count if max_count>0 else np.nan

    # Estimasi TINN (approx)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if counts.size >= 3 and counts.max() > 0:
        idx_peak = np.argmax(counts)
        left_nonzero = np.where(counts[:idx_peak] > 0)[0]
        right_nonzero = np.where(counts[idx_peak+1:] > 0)[0] + (idx_peak+1)
        if left_nonzero.size>0:
            left_idx0 = left_nonzero[0]
        else:
            left_idx0 = 0
        if right_nonzero.size>0:
            right_idx0 = right_nonzero[-1]
        else:
            right_idx0 = len(counts)-1

        x_peak = bin_centers[idx_peak]; y_peak = counts[idx_peak]
        x_left = bin_centers[left_idx0]; y_left = counts[left_idx0]
        x_right = bin_centers[right_idx0]; y_right = counts[right_idx0]

        def intercept_x(x1,y1,x2,y2):
            if y2 == y1:
                return (x1 + x2) / 2.0
            return x1 - y1 * (x2 - x1) / (y2 - y1)

        x_left_int = intercept_x(x_peak, y_peak, x_left, y_left)
        x_right_int = intercept_x(x_peak, y_peak, x_right, y_right)
        tinn = x_right_int - x_left_int
    else:
        tinn = np.nan

    return {
        "N_intervals": N,
        "mean_RR_s": mean_rr,
        "mean_HR_bpm": mean_hr,
        "SDNN_s": sdnn,
        "SDANN_s": sdann,
        "SDNN_index_s": sdnn_index,
        "RMSSD_s": rmssd,
        "SDSD_s": sdsd,
        "NN50_count": int(nn50),
        "pNN50_percent": float(pnn50),
        "HTI": float(hti),
        "TINN_s_approx": float(tinn),
        "CVNN_percent": float(cvnn),
        "CVSD_percent": float(cvsd),
        "Skewness": float(skewness)
    }
    
def hrv_frequency_domain(rr_intervals, fs=4.0):
    """
    Compute HRV frequency domain features using Welch's method.
    rr_intervals: array of RR intervals (in seconds)
    fs: interpolation frequency (Hz), typically 4 Hz for HRV analysis
    """

    # --- 1. Interpolate RR intervals to get evenly spaced signal ---
    time_rr = np.cumsum(np.concatenate(([0], rr_intervals)))  # Start from 0
    t_interp = np.arange(0, time_rr[-1], 1/fs)
    rr_interp = np.interp(t_interp, time_rr[1:], rr_intervals)  # Use time_rr[1:] to match rr_intervals length

    # --- 2. Apply Welch PSD estimation ---
    f, psd = welch(rr_interp, fs=fs, nperseg=min(256, len(rr_interp)))

    # --- 3. Define HRV frequency bands ---
    vlf_band = (0.003, 0.04)
    lf_band  = (0.04, 0.15)
    hf_band  = (0.15, 0.40)

    # --- 4. Integrate power in each band ---
    def band_power(band):
        mask = (f >= band[0]) & (f <= band[1])
        return np.trapz(psd[mask], f[mask]) if np.any(mask) else 0.0

    vlf_power = band_power(vlf_band)
    lf_power = band_power(lf_band)
    hf_power = band_power(hf_band)
    total_power = vlf_power + lf_power + hf_power

    # --- 5. Normalized units ---
    lf_nu = (lf_power / (total_power - vlf_power)) * 100 if (total_power - vlf_power) > 0 else 0
    hf_nu = (hf_power / (total_power - vlf_power)) * 100 if (total_power - vlf_power) > 0 else 0

    # --- 6. LF/HF ratio ---
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

    # --- 7. Peak frequencies ---
    lf_peak = f[np.argmax(psd[(f >= lf_band[0]) & (f <= lf_band[1])])] if np.any((f >= lf_band[0]) & (f <= lf_band[1])) else 0
    hf_peak = f[np.argmax(psd[(f >= hf_band[0]) & (f <= hf_band[1])])] if np.any((f >= hf_band[0]) & (f <= hf_band[1])) else 0

    return {
        "TP": total_power,
        "LF Power": lf_power,
        "HF Power": hf_power,
        "LF/HF Ratio": lf_hf_ratio,
        "LF (n.u.)": lf_nu,
        "HF (n.u.)": hf_nu,
        "LF Peak Freq": lf_peak,
        "HF Peak Freq": hf_peak
    }

def hrv_nonlinear(rr_intervals, show_plot=False):
    """
    Compute non-linear HRV parameters using Poincaré plot (SD1, SD2, SD1/SD2).
    rr_intervals: array of RR intervals (in seconds)
    """
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    # Calculate mean of RR intervals
    rr_mean = np.mean(rr_intervals)

    # Poincaré coordinates
    diff_rr = rr_n1 - rr_n

    # Compute SD1 and SD2
    sd1 = np.sqrt(np.var(diff_rr) / 2.0)
    sd2 = np.sqrt(2 * np.var(rr_intervals) - (np.var(diff_rr) / 2.0))
    sd_ratio = sd1 / sd2 if sd2 != 0 else np.inf

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(rr_n, rr_n1, alpha=0.6, color='blue', label='RRn vs RRn+1')
        plt.title('Poincaré Plot')
        plt.xlabel('RRn (s)')
        plt.ylabel('RRn+1 (s)')
        plt.plot([rr_mean - 0.5, rr_mean + 0.5],
                 [rr_mean - 0.5, rr_mean + 0.5], 'r--', label='Line of Identity')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        "SD1": sd1,
        "SD2": sd2,
        "SD1/SD2": sd_ratio
    }
