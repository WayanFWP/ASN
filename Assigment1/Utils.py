import numpy as np
import ctypes
import matplotlib.pyplot as plt

class Complex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

# Load .so
convo = ctypes.CDLL("Acceleration/convolution.so")
fft = ctypes.CDLL("Acceleration/FFT.so")

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
    
    # # y[n] = x[nM]
    # y = np.array([x[n * M] for n in range(len(x) // M)])
    # return y
    
    arr = np.asarray(x)
    
    # If M is integer (or very close), use fast decimation by indexing
    if float(M).is_integer():
        M_int = int(M)
        if M_int == 1:
            return arr.copy()
        return arr[::M_int]
    
    # For non-integer M (float), perform linear resampling:
    # new_length = floor(len(x) / M)  (if M>1 -> downsample, if 0<M<1 -> upsample)
    new_len = max(1, int(np.floor(len(arr) / float(M))))
    old_pos = np.arange(len(arr))
    new_pos = np.linspace(0, len(arr) - 1, new_len)
    y = np.interp(new_pos, old_pos, arr).astype(arr.dtype)
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

from scipy.signal import butter, filtfilt

def BPF(x, fc, fh, fs):
    nyqs = fs / 2
    low = fc / nyqs 
    high = min(fh, nyqs - 1) / nyqs

    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, x)

def triangle(t, a, k, b, Y):
    q = np.zeros_like(t)
    left_mask = (t >= t[a]) & (t <= t[k])
    right_mask = (t >= t[k]) & (t <= t[b])

    if t[k] != t[a]:
        q[left_mask] = Y * (t[left_mask] - t[a]) / (t[k] - t[a])
    if t[b] != t[k]:
        q[right_mask] = Y * (t[b] - t[right_mask]) / (t[b] - t[k])
    return q

