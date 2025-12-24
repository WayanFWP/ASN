# PPG Multiscale Decomposer

A comprehensive signal analysis tool for Photoplethysmography (PPG) signals using multi-scale decomposition via Discrete Wavelet Transform (DWT). This project extracts physiological information including heart rate, respiratory rate, vasometric activity, and detailed Heart Rate Variability (HRV) metrics.

## Features

- **Multi-Scale Signal Decomposition**: Custom DWT implementation with 8 decomposition levels
- **Heart Rate Analysis**: Peak detection and BPM calculation
- **Respiratory Analysis**: Breathing rate estimation from PPG signals
- **Vasometric Analysis**: Low-frequency vasomotor activity detection
- **Comprehensive HRV Analysis**:
  - Time-domain metrics (SDNN, RMSSD, pNN50, etc.)
  - Frequency-domain metrics (VLF, LF, HF power)
  - Nonlinear metrics (Poincaré plot analysis)
- **C-Accelerated Computing**: Fast FFT and convolution using C libraries
- **Multiple Interfaces**:
  - Command-line interface (main.py)
  - Streamlit web application
  - Tkinter desktop GUI
- **Interactive Visualizations**: Real-time plotting and analysis

## Architecture

```
ppg-multiscale-decomposer/
├── main.py                    # CLI entry point
├── streamlit_app.py           # Web interface
├── Coeficient.py              # DWT filter coefficient generator
├── analysis.py                 # Signal analysis modules
├── HRV.py                     # HRV feature extraction
├── Utils.py                   # Utility functions and C bindings
├── experiment2language.py     # FFT testing script
└── Acceleration/              # C acceleration modules
    ├── convolution.c          # Fast convolution
    └── FFT.c                  # Fast Fourier Transform
```

## Installation

### Prerequisites

- Python 3.8+
- GCC compiler (for C acceleration modules)
- NumPy, SciPy, Matplotlib, Pandas

### Step 1: Clone the Repository

```bash
git clone https://github.com/WayanFWP/ASN
cd ASN/ppg-multiscale-decomposer
```

### Step 2: Install Python Dependencies

```bash
pip install numpy scipy matplotlib pandas streamlit plotly
```

### Step 3: Compile C Libraries

#### On Linux/macOS: 
```bash
cd Acceleration
gcc -shared -fPIC -o convolution.so convolution.c
gcc -shared -fPIC -o FFT.so FFT.c -lm
cd ..
```
notes: this might not works, because my current linux setup is doomed so stay tune with the update

#### On Windows (MinGW):
```bash
cd Acceleration
gcc -shared -o FFT.dll FFT.c -lm -static-libgcc
gcc -shared -o convolution.dll convolution.c -static-libgcc
cd ..
```

### Step 4: Prepare Your Data

Place your PPG signal data in CSV format in a `data/` directory. Ensure your CSV has:
- Time or sample index column
- PPG signal column(s)

## Usage

### Command Line Interface

```python
python main.py
```

Edit the following parameters in `main.py`:
```python
load_file = pd.read_csv("data/your_data.csv")
fs = 50              # Sampling frequency in Hz
factor = 1.15        # Downsampling factor
selected_signal = load_file.columns[1]  # Signal column
```

### Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Features:
- Upload CSV files through web interface
- Adjust parameters in real-time
- Interactive plots with Plotly
- Export results and visualizations


Features:
- Native desktop application
- File browser for data selection
- Progress tracking
- Export plots as PNG/PDF

## Modules

### 1. Coeficient.py - DWT Filter Bank

Implements custom Discrete Wavelet Transform filter bank:

```python
from Coeficient import Coeficient

coef = Coeficient(fs=50)  # Initialize with sampling frequency
coef.initialize_qj_filter()  # Create filter bank

# Apply specific decomposition level
signal_dwt = coef.applying(signal, specific_j=7)
```

**Key Methods:**
- `initialize_qj_filter()`: Generates Q_j filters for j=1 to j=8
- `applying(signal, specific_j=None)`: Apply DWT at specific level(s)
- `plot_filter_responses()`: Visualize filter frequency responses
- `getAnBvalues(j)`: Compute filter support range for level j

### 2. analysis.py - Physiological Signal Analysis

#### HeartRate Class
```python
from analysis import HeartRate

hr = HeartRate(fs=50)
signal_hr, peaks, bpm = hr.analysis(ppg_signal)
print(f"BPM: {bpm:.2f}")
```

#### Respiratory Class
```python
from analysis import Respiratory

resp = Respiratory(fs=50)
signal_resp, peaks, brpm = resp.analysis(respiratory_signal)
freq, magnitude, peak_freq, peak_mag = resp.get_freq()
print(f"Respiratory Rate: {brpm:.2f} breaths/min")
```

#### Vasometric Class
```python
from analysis import Vasometric

vaso = Vasometric(fs=50)
freq, magnitude, peak_freq, peak_mag = vaso.analysis(vasometric_signal)
print(f"Vasomotor Frequency: {peak_freq:.4f} Hz")
```

### 3. HRV.py - Heart Rate Variability Analysis

Comprehensive HRV feature extraction across three domains:

#### Time Domain Features
```python
from HRV import HRV

rr_intervals = np.diff(peaks) / fs  # RR intervals in seconds
hrv = HRV(rr_intervals)
time_features = hrv.compute_all()['time']

# Available metrics:
# - SDNN: Standard deviation of NN intervals
# - RMSSD: Root mean square of successive differences
# - pNN50: Percentage of successive differences > 50ms
# - SDSD: Standard deviation of successive differences
# - HTI: HRV triangular index
# - TINN: Triangular interpolation of NN interval histogram
```

#### Frequency Domain Features
```python
freq_features = hrv.compute_all()['frequency']

# Available metrics:
# - VLF Power: Very Low Frequency (0.003-0.04 Hz)
# - LF Power: Low Frequency (0.04-0.15 Hz)
# - HF Power: High Frequency (0.15-0.4 Hz)
# - LF/HF Ratio: Sympatho-vagal balance indicator
# - Total Power: Total spectral power
```

#### Nonlinear Domain Features
```python
nonlinear_features = hrv.compute_all()['nonlinear']

# Available metrics:
# - SD1: Short-term HRV (Poincaré plot)
# - SD2: Long-term HRV (Poincaré plot)
# - SD1/SD2 Ratio: Ratio of short to long-term variability
```

### 4. Utils.py - Utility Functions

Core signal processing utilities:

```python
from Utils import *

# Downsampling
downsampled = downSample(signal, factor=2.0)

# Bandpass filtering
filtered = BPF(signal, fc=1, fh=45, fs=50)

# FFT (C-accelerated)
magnitude, freq = FFT(signal, fs=50)

# Convolution (C-accelerated)
result = convolve(signal, filter_coeffs)
```

## Signal Processing Pipeline

The typical analysis pipeline:

```python
# 1. Load and preprocess
signal = load_file['PPG'].values
signal = downSample(signal, factor=1.15)
signal = signal - np.mean(signal)  # Remove DC offset
signal = BPF(signal, 1, 45, fs)    # Bandpass 1-45 Hz

# 2. Initialize analysisrs
coef = Coeficient(fs)
coef.initialize_qj_filter()
hr_analysisr = HeartRate(fs)
resp_analysisr = Respiratory(fs)
vaso_analysisr = Vasometric(fs)

# 3. Heart rate analysis (original signal)
signal_hr, peaks, bpm = hr_analysisr.analysis(signal)
rr_intervals = np.diff(peaks) / fs

# 4. Respiratory analysis (DWT level 7)
signal_dwt7 = coef.applying(signal, specific_j=7)
resp_signal, resp_peaks, brpm = resp_analysisr.analysis(signal_dwt7[7])

# 5. Vasometric analysis (DWT level 8)
signal_dwt8 = coef.applying(signal, specific_j=8)
vaso_freq, vaso_mag, peak_vaso, _ = vaso_analysisr.analysis(signal_dwt8[8])

# 6. HRV analysis
hrv = HRV(rr_intervals)
hrv_features = hrv.compute_all()
```

## Output Metrics

### Heart Rate Metrics
- **BPM**: Beats per minute
- **RR Intervals**: Time between consecutive heartbeats
- **Peak Locations**: Systolic peak positions

### Respiratory Metrics
- **BrPM**: Breaths per minute
- **Respiratory Frequency**: Dominant frequency in Hz
- **Peak Magnitude**: Amplitude of respiratory component

### Vasometric Metrics
- **Vasomotor Frequency**: Dominant low-frequency oscillation
- **Peak Magnitude**: Amplitude of vasomotor component

### HRV Metrics

#### Time Domain
| Metric | Description | Units | Normal Range |
|--------|-------------|-------|--------------|
| SDNN | Standard deviation of NN intervals | ms | 50-100 ms |
| RMSSD | Root mean square of successive differences | ms | 20-50 ms |
| pNN50 | % of successive differences > 50ms | % | 5-50% |
| CVNN | Coefficient of variation | % | - |
| SDSD | SD of successive differences | ms | - |
| HTI | HRV triangular index | - | 20-50 |
| TINN | Triangular interpolation | ms | 200-400 ms |

#### Frequency Domain
| Metric | Description | Units | Normal Range |
|--------|-------------|-------|--------------|
| VLF Power | Very low frequency power | ms² | - |
| LF Power | Low frequency power | ms² | 200-1000 ms² |
| HF Power | High frequency power | ms² | 200-1000 ms² |
| LF/HF Ratio | Sympatho-vagal balance | - | 1-3 |
| Total Power | Total spectral power | ms² | - |

#### Nonlinear Domain
| Metric | Description | Units |
|--------|-------------|-------|
| SD1 | Short-term variability | ms |
| SD2 | Long-term variability | ms |
| SD1/SD2 | Ratio | - |

## Depedencies

### Python Dependencies
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
streamlit>=1.0.0  # For web app
plotly>=5.0.0     # For interactive plots
```

### System Requirements
- GCC/MinGW for compiling C libraries
- Python 3.8 or higher

## Examples

### Example 1: Quick Analysis

```python
from Coeficient import Coeficient
from analysis import HeartRate, Respiratory, Vasometric
from HRV import HRV
from Utils import *
import pandas as pd

# Load data
df = pd.read_csv("data/ppg_signal.csv")
fs = 50
signal = downSample(df['PPG'].values, 1.15)
signal = BPF(signal - np.mean(signal), 1, 45, fs/1.15)

# Setup
coef = Coeficient(fs/1.15)
coef.initialize_qj_filter()

# analysis
hr = HeartRate(fs/1.15)
_, peaks, bpm = hr.analysis(signal)
print(f"Heart Rate: {bpm:.1f} BPM")

# HRV
rr = np.diff(peaks) / (fs/1.15)
hrv = HRV(rr)
features = hrv.compute_all()
print(f"SDNN: {features['time']['SDNN']:.1f} ms")
```

### Example 2: Custom DWT Analysis

```python
from Coeficient import Coeficient
import numpy as np

# Generate test signal
fs = 50
t = np.arange(0, 60, 1/fs)
signal = np.sin(2*np.pi*1.2*t) + 0.5*np.sin(2*np.pi*0.3*t)

# Apply multi-level decomposition
coef = Coeficient(fs)
coef.initialize_qj_filter()

# Get all levels
all_levels = coef.applying(signal)

# Get specific level
level_7 = coef.applying(signal, specific_j=7)
```

### Example 3: Batch Processing

```python
import glob
import pandas as pd

results = []
for file in glob.glob("data/*.csv"):
    df = pd.read_csv(file)
    signal = downSample(df['PPG'].values, 1.15)
    signal = BPF(signal - np.mean(signal), 1, 45, fs)
    
    hr = HeartRate(fs)
    _, peaks, bpm = hr.analysis(signal)
    
    results.append({
        'file': file,
        'bpm': bpm,
        'n_beats': len(peaks)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("batch_results.csv", index=False)
```

## Theory Background

### Discrete Wavelet Transform

The project implements a custom DWT filter bank based on iterative filter design:

**Level j filter Q_j is computed as:**
- Q₁[k] = -2(δ[k] - δ[k+1])
- Q_j[k] = Q_{j-1}[k] x h[k] for j > 1

Where:
- h[k] is the scaling function: [0, 1/8, 3/8, 3/8]
- x denotes convolution
- k ranges from a_j to b_j where:
  - a_j = -(2^j + 2^{j-1} - 2)
  - b_j = -(1 - 2^{j-1}) + 1

### Physiological Signal Separation

- **Heart Rate**: analysisd from original signal (1-45 Hz)
- **Respiratory**: Extracted at DWT level 7 (0.15-0.4 Hz)
- **Vasometric**: Extracted at DWT level 8 (0.04-0.15 Hz)

### HRV Analysis

The HRV module follows Task Force guidelines (1996) for computing standard metrics with modern enhancements for robust estimation.

## Acknowledgments

- Based on wavelet decomposition theory from signal processing literature
- HRV computation follows Task Force of ESC and NASPE guidelines
- C acceleration inspired by high-performance computing practices

---

**Note**: This tool is for research and educational purposes. Not intended for clinical diagnosis.
