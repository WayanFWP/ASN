# CWT-STFT Analysis for PCG and ECG Signals

## Overview

This project implements signal processing techniques for analyzing Phonocardiogram (PCG) and Electrocardiogram (ECG) signals. It uses Continuous Wavelet Transform (CWT) and Short-Time Fourier Transform (STFT) to detect and analyze heart sounds (S1 and S2) based on ECG R-peak detection.

## Features

- **R-Peak Detection**: Implements Pan-Tompkins algorithm for ECG R-peak detection
- **Interactive Segmentation**: Visual selection of R-peaks for PCG segmentation (±500ms windows)
- **CWT Analysis**: Continuous Wavelet Transform using Morlet wavelets for time-frequency analysis
- **Heart Sound Detection**: Automatic detection of S1 and S2 heart sounds using energy-based thresholding and connected component analysis
- **STFT Analysis**: Short-Time Fourier Transform with customizable window functions
- **Visualization**: Comprehensive plotting of signals, scalograms, and spectrograms

## Project Structure

```
CWT-STFT_analysis/
├── main.py          # Main execution script
├── Utils.py         # Core signal processing functions
├── plot.py          # Visualization functions
├── preproceed.py    # Data preprocessing and format conversion
├── README.md        # This file
└── dat/             # Data directory
    ├── a0003.csv
    ├── a0004.csv
    ├── a0007.csv
    ├── a0014.csv
    └── a0042.csv
```

## Requirements

```python
numpy
pandas
scipy
matplotlib
wfdb  
```

## Installation

1. Clone or download the project
2. Install required packages:
```bash
pip install numpy pandas scipy matplotlib wfdb
```

## Usage

### Basic Workflow

1. **Select a data file** in [main.py](main.py):
```python
namefile = "a0007"  # Change this to your desired file
```

2. **Run the main script**:
```bash
python main.py
```

3. **Interactive Steps**:
   - The script will display detected R-peaks in the ECG signal
   - Select an R-peak index when prompted
   - The script will segment the PCG signal around the selected R-peak (±500ms)
   - CWT and STFT analyses will be performed automatically
   - Results will be visualized in multiple plots

### Data Preprocessing

To convert raw WFDB format to CSV:

1. Place raw `.hea` and `.dat` files in `dat/raw/`
2. Edit [preproceed.py](preproceed.py):
```python
namefile = "a0042"  # Your file name
```
3. Run:
```bash
python preproceed.py
```

## Algorithms

### 1. Pan-Tompkins R-Peak Detection

The Pan-Tompkins algorithm detects QRS complexes in ECG signals through:

- **Bandpass filtering** (5-15 Hz) to reduce noise
- **Derivative filter** to emphasize QRS complex slopes
- **Squaring** to intensify QRS detection
- **Moving window integration** (~150ms) to smooth the signal
- **Adaptive thresholding** for peak detection

Implementation in [Utils.py](Utils.py#L54):
```python
filtered, r_peaks = pan_tompkins(ECG_data.values, fs)
```

### 2. Continuous Wavelet Transform (CWT)

CWT decomposes the signal into time-frequency components using Morlet wavelets:

$$\psi(t) = \pi^{-1/4} e^{j\omega_0 t} e^{-t^2/2}$$

Key features:
- **Center frequency**: 0.849 Hz (Delphi convention)
- **Scale range**: 2ms to 100ms (logarithmic spacing)
- **Normalization**: $1/\sqrt{s}$ for energy preservation

The CWT is computed as:

$$W(s,\tau) = \frac{1}{\sqrt{s}} \int x(t) \psi^*\left(\frac{t-\tau}{s}\right) dt$$

### 3. Heart Sound Detection

S1 and S2 detection uses a multi-step approach:

1. **Energy Computation**: $E(s,t) = |W(s,t)|^2$
2. **Time Windows**:
   - S1: -100ms to +200ms relative to R-peak
   - S2: +200ms to +400ms relative to R-peak
3. **Thresholding**:
   - S1 threshold: 60% of maximum energy
   - S2 threshold: 15% of maximum energy (configurable)
4. **Connected Components**: Flood-fill algorithm to identify regions
5. **Center of Gravity (CoG)**: Weighted centroid of each region

$$t_{CoG} = \frac{\sum_i t_i E_i}{\sum_i E_i}, \quad s_{CoG} = \frac{\sum_j s_j E_j}{\sum_j E_j}$$

### 4. Short-Time Fourier Transform (STFT)

STFT provides time-frequency representation with fixed resolution:

$$X(f,\tau) = \int x(t) w(t-\tau) e^{-j2\pi ft} dt$$

Features:
- **Window functions**: Hanning, Hamming, Triangular, Rectangular
- **Configurable parameters**:
  - `window_size`: Number of samples per frame (default: 512)
  - `hop_size`: Frame overlap in samples (default: 64)
- **Energy normalization** for consistent visualization

## Configuration Parameters

### In [main.py](main.py)

```python
# File selection
namefile = "a0007"

# Sampling frequency
fs = 2000  # Hz

# Heart sound detection thresholds
s1, s2 = 0.7, 0.05  # Relative to max energy

# STFT parameters
window_size = 512
hop_size = 64
scale = False
```

### R-Peak Segmentation

- **Window size**: ±500ms around R-peak
- **Expected S1 region**: -200ms to +100ms from R-peak
- **Expected S2 region**: +100ms to +400ms from R-peak

## Visualization Outputs

### 1. ECG with R-Peaks
- Full ECG signal with detected R-peaks
- PCG signal with ±500ms windows highlighted
- Color-coded R-peak selection guide

### 2. PCG Segmentation
- Full signal with selected segment
- Zoomed segment in absolute time
- Segment centered on R-peak (relative time)
- Expected S1/S2 regions highlighted

### 3. CWT Scalogram
- Time-frequency energy distribution
- S1 and S2 masks overlaid as contours
- Center of Gravity points marked
- Normalized energy colormap

### 4. STFT Spectrogram
- Time-frequency magnitude representation
- dB scale for better dynamic range
- Positive frequencies only

## Troubleshooting

### No R-peaks detected
- Check ECG signal quality
- Adjust bandpass filter parameters in `pan_tompkins()`
- Lower the threshold in `find_peaks()`

### S2 not detected
- Increase S2 threshold: `s2 = 0.05` → `s2 = 0.1`
- Check if S2 falls within expected time window
- Verify PCG segment quality

### Poor CWT resolution
- Adjust scale range in `cwt_analysis()`:
```python
MAX_SCALE_SEC = 0.1  # Increase for lower frequencies
MIN_SCALE_SEC = 0.002  # Decrease for higher frequencies
```

## Data Format

### Input CSV Format
```
Time, PCG, ECG
0.0000, 0.123, 0.456
0.0005, 0.124, 0.457
...
```

- **Time**: Seconds (float)
- **PCG**: Phonocardiogram amplitude (float)
- **ECG**: Electrocardiogram amplitude (float)
- **Sampling frequency**: 2000 Hz

## References

1. **Pan-Tompkins Algorithm**: Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE transactions on biomedical engineering*, (3), 230-236.

2. **Wavelet Transform**: Addison, P. S. (2002). *The illustrated wavelet transform handbook: introductory theory and applications in science, engineering, medicine and finance*. CRC press.

3. **Heart Sound Detection**: Springer, D. B., et al. (2016). Logistic regression-HSMM-based heart sound segmentation. *IEEE Transactions on Biomedical Engineering*, 63(4), 822-832.
