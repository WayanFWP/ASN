import numpy as np
import ctypes

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