import numpy as np
import ctypes

# Load .so
lib = ctypes.CDLL("/home/nayaw/project/ASN/Assigment1/convolution.so")

# Your C function signature: double* convolution(const double*, int, const double*, int)
lib.convolution.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]
lib.convolution.restype = ctypes.POINTER(ctypes.c_double)  # returns double*

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
    
    # Call the C function
    result_ptr = lib.convolution(
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), signal_len,
        filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), filt_len
    )
    
    # Convert result back to numpy array
    result = np.array([result_ptr[i] for i in range(conv_len)])
    
    # Free the allocated memory
    ctypes.CDLL('libc.so.6').free(result_ptr)
    
    return result