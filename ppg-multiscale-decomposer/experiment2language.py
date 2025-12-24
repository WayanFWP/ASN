import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load library
lib = ctypes.CDLL('./FFT.so')
class Complex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]


# === DEFINISI ARGUMEN ===
lib.fft.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
                    ctypes.c_int]
lib.fft.restype = ctypes.POINTER(Complex128)

lib.free_memory.argtypes = [ctypes.c_void_p]
lib.free_memory.restype = None

# --- Parameter ---
fs = 8.33
N = 4000

# Contoh sinyal
t = np.arange(N) / fs
x = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t)
x = x - np.mean(x)  # Hilangin DC offset

# --- FFT call ---
x_complex = np.array(x, dtype=np.complex128)
res_ptr = lib.fft(x_complex, N)
result = np.ctypeslib.as_array(res_ptr, shape=(N,)).view(np.complex128)
lib.free_memory(res_ptr)

# --- Frequency & Magnitude ---
magnitude = np.abs(result[:N//2]) / (N/2)
freq = np.arange(N//2) * fs / N

# --- Plot ---
plt.plot(freq, magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.grid(True)
plt.show()
