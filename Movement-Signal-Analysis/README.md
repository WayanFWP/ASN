# Movement Signal Analysis

A comprehensive EMG (Electromyography) signal processing and analysis toolkit for gait cycle detection and muscle activation analysis.

## Overview

This project provides tools for analyzing EMG signals from lower limb muscles during gait cycles. It processes signals from:
- **Gastrocnemius Lateralis (GL)** - lateral calf muscle
- **Vastus Lateralis (VL)** - lateral thigh muscle
- **Foot Switch** - ground contact detection

The toolkit implements advanced signal processing techniques including:
- Digital filtering (LPF, HPF, BPF)
- Discrete Wavelet Transform (DWT) denoising
- Continuous Wavelet Transform (CWT) time-frequency analysis
- Short-Time Fourier Transform (STFT)
- Automatic onset/offset detection for muscle activations

## Features

- Gait cycle segmentation based on foot-switch data
- Multi-scale wavelet filtering with customizable Q-factor
- CWT-based muscle activation detection
- Adaptive threshold algorithms (Teager Energy, RMS, Mean, Percentile)
- Interactive visualization with matplotlib
- Comprehensive statistical analysis
- Support for multiple gait cycles

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib scipy wfdb
```

### Dependencies

- **numpy** - Numerical computations
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **scipy** - Signal processing utilities
- **wfdb** - PhysioNet waveform database tools

## Project Structure

```
Movement-Signal-Analysis/
├── Filter.py          # Signal filtering classes (LPF, HPF, BPF, CWT, STFT, DWT)
├── main.py            # Main execution script
├── plot.py            # Visualization functions
├── preprocess.py      # Data extraction from WFDB format
├── utils.py           # Helper functions for detection and segmentation
└── data/
    └── S01_extracted.csv  # Preprocessed EMG data
```

## Usage

### 1. Data Preprocessing

Extract signals from WFDB format:

```python
python preprocess.py
```

This script:
- Reads raw `.dat` and `.hea` files from PhysioNet database
- Extracts foot switch, GL, and VL signals
- Saves to CSV format for analysis

### 2. Main Analysis Pipeline

Run the complete analysis:

```python
python main.py
```

#### Interactive Workflow:

1. **Select Q-factor** for DWT filtering (suggested: 4)
2. **Choose visualization options**:
   - Full dataset plot
   - Process all segments or select specific ones
   - Show CWT scalograms
   - Show detection plots

3. **Review results**:
   - Muscle activation timings
   - Statistical summaries
   - STFT analysis

### Example Output

```
Found 15 toe-off events at indices: [1234, 2456, 3678, ...]
Created 14 gait cycle segments

Processing Segment 0 (1/14)
==================================================
  GL - Found 1 activation(s)
    Activation 1: 0.123s to 0.567s (duration: 0.444s)
  
  VL - Found 1 activation(s)
    Activation 1: 0.089s to 0.432s (duration: 0.343s)

FINAL RESULTS ACROSS ENTIRE DATASET
==================================================
Total segments processed: 14
Total GL activations found: 14
Total VL activations found: 13

GL activation rate: 14/14 segments (100.0%)
VL activation rate: 13/14 segments (92.9%)
```

## Module Documentation

### Filter.py

#### `class Filter(fs)`

Main filtering class for EMG signal processing.

**Parameters:**
- `fs` (int): Sampling frequency in Hz

**Methods:**

##### `LPF(signal, cutoff)`
Second-order Butterworth low-pass filter.
- **signal**: Input signal array
- **cutoff**: Cutoff frequency in Hz
- **Returns**: Filtered signal

##### `HPF(signal, cutoff)`
Second-order Butterworth high-pass filter.
- **signal**: Input signal array
- **cutoff**: Cutoff frequency in Hz
- **Returns**: Filtered signal

##### `BPF(signal, lowcut, highcut)`
Band-pass filter combining HPF and LPF.
- **signal**: Input signal array
- **lowcut**: Lower cutoff frequency
- **highcut**: Upper cutoff frequency
- **Returns**: Filtered signal

##### `cwt(signal, scales, percentage_keep=None)`
Continuous Wavelet Transform using Morlet wavelet.
- **signal**: Input signal
- **scales**: Array of scales
- **percentage_keep**: Keep top percentage of coefficients (compression)
- **Returns**: (cwt_matrix, frequencies)

##### `cwt_analysis(segmented_signals, percentage_keep=1)`
Complete CWT analysis for a signal segment.
- **segmented_signals**: Signal segment to analyze
- **percentage_keep**: Coefficient retention ratio (0-1)
- **Returns**: (coefficients, frequencies, time_axis)

##### `stft(x, window_size, hop_size, window='hann', percentage=None)`
Short-Time Fourier Transform implementation.
- **x**: Input signal
- **window_size**: Window length in samples
- **hop_size**: Hop size in samples
- **window**: Window type ('hann', 'hamming', 'triangular')
- **percentage**: Keep top percentage of coefficients
- **Returns**: STFT matrix

#### `class DWT(fs)`

Discrete Wavelet Transform for multi-resolution denoising.

**Parameters:**
- `fs` (int): Sampling frequency

**Methods:**

##### `applying(signal, specific_j=None)`
Apply DWT filtering at specified resolution level(s).
- **signal**: Input signal
- **specific_j**: Resolution level (1-8) or None for all
- **Returns**: Dictionary of filtered signals by level

##### `initialize_qj_filter()`
Initialize quadrature mirror filters for DWT decomposition.

##### `plot_filter_responses(fig=None)`
Visualize frequency responses of DWT filters.

### utils.py

#### Core Functions

##### `detect_cycle(data, threshold=0.2, debounce_len=30)`
Detect heel-strike and toe-off events from foot-switch signal.
- **data**: Foot-switch signal
- **threshold**: Binary threshold (0-1)
- **debounce_len**: Minimum samples between events
- **Returns**: List of (heel_strike, toe_off, next_heel_strike) tuples

##### `extract_toe_off_events(data, threshold=0.2, debounce_len=30)`
Extract only toe-off events for segmentation.
- **Returns**: Array of toe-off indices

##### `segment_gait(signals, toe_off_events)`
Segment signals into individual gait cycles.
- **signals**: Dict of EMG signals {'gl': array, 'vl': array}
- **toe_off_events**: Array of toe-off indices
- **Returns**: List of segment dictionaries

##### `compute_cwt_energy(cwt_coeffs, frequencies, freq_range=(20, 200))`
Compute energy envelope from CWT coefficients.
- **cwt_coeffs**: CWT coefficient matrix
- **frequencies**: Frequency array
- **freq_range**: Frequency band of interest (Hz)
- **Returns**: Energy envelope array

##### `adaptive_threshold(signal, method='teager', factor=2.0, baseline_percentage=0.2)`
Compute adaptive threshold for activation detection.
- **signal**: Energy envelope
- **method**: 'teager', 'rms', 'mean', or 'percentile'
- **factor**: Multiplier for threshold
- **baseline_percentage**: Portion of signal for baseline
- **Returns**: Threshold value

##### `detect_onset_offset_cwt(cwt_coeffs, frequencies, fs, ...)`
Detect single muscle activation onset and offset.
- **cwt_coeffs**: CWT coefficients
- **frequencies**: Frequency array
- **fs**: Sampling frequency
- **method**: Threshold method
- **threshold_factor**: Threshold multiplier
- **min_duration**: Minimum activation duration (seconds)
- **freq_range**: Frequency band (Hz)
- **merge_window**: Merge nearby activations (seconds)
- **Returns**: (onset_idx, offset_idx, energy_envelope, threshold)

##### `detect_multiple_activations_cwt(cwt_coeffs, frequencies, fs, ...)`
Detect multiple muscle activations in a segment.
- **Returns**: (activations_list, energy_envelope, threshold)

##### `merge_close_activations(activations, merge_window_seconds, fs, debug=True)`
Merge activations separated by less than merge window.
- **activations**: List of (onset, offset) tuples
- **merge_window_seconds**: Merge threshold (seconds)
- **Returns**: Merged activation list

### plot.py

#### Visualization Functions

##### `plot_signals(data, fs, cycles=None)`
Plot foot switch, GL, and VL signals with gait cycle markers.

##### `plot_scalogram(cwt_matrix, freqs, time, title)`
Display CWT scalogram (time-frequency representation).

##### `plot_combined_scalograms(cwt_gl, freqs_gl, time_gl, cwt_vl, freqs_vl, time_vl, segment_idx, ...)`
Show GL and VL scalograms side-by-side.

##### `plot_tabbed_detection_debug(...)`
Interactive tabbed plot showing detection results for GL and VL.

##### `plot_comprehensive_analysis(full_time, filtered, cycles, segments_to_process, ...)`
Generate final comprehensive analysis figure with all results.

##### `plot_stft_result(stft_matrix, frequencies, time_frames, fs=1, db_scale=True)`
Visualize STFT spectrogram.

## Algorithm Details

### Muscle Activation Detection Pipeline

1. **Preprocessing**
   - Band-pass filtering (20-200 Hz typical for EMG)
   - DWT denoising with Q-factor selection

2. **Time-Frequency Analysis**
   - CWT using Morlet wavelet
   - Frequency range: 20-200 Hz (adjustable)
   - Scales: logarithmically spaced

3. **Energy Computation**
   - Sum magnitude across frequency band
   - Produces 1D energy envelope

4. **Threshold Detection**
   - Adaptive threshold methods:
     - **Teager Energy**: Nonlinear operator sensitive to instantaneous energy
     - **RMS**: Root mean square of baseline
     - **Mean**: Simple mean-based threshold
     - **Percentile**: Distribution-based threshold

5. **Activation Extraction**
   - Find threshold crossings
   - Apply minimum duration filter
   - Merge close activations
   - Return onset/offset times

### DWT Filtering

The project uses a custom DWT implementation with quadrature mirror filters:

- **Decomposition levels**: Q = 1 to 8
- **Higher Q**: More aggressive smoothing
- **Recommended Q = 4**: Balance between noise removal and signal preservation

## Configuration

### Key Parameters in main.py

```python
fs = 2000              # Sampling frequency (Hz)
Q = 4                  # DWT decomposition level
window_size = 64       # STFT window size (ms)
```

### Detection Parameters

```python
# In utils.py functions:
freq_range=(20, 200)        # EMG frequency band
threshold_factor=2.0        # Threshold multiplier
min_duration=0.05          # Minimum activation duration (s)
merge_window=0.2           # Merge window (s)
method='teager'            # Threshold method
```

## Data Format

### Input (WFDB Format)

- `.dat` - Binary signal data
- `.hea` - Header with metadata

### Processed CSV Format

| foot_switch | gl | vl |
|-------------|----|----|
| 0.123 | 0.045 | 0.067 |
| 0.145 | 0.052 | 0.071 |
| ... | ... | ... |

- **foot_switch**: Pressure sensor (0-1 normalized)
- **gl**: Gastrocnemius Lateralis EMG (mV)
- **vl**: Vastus Lateralis EMG (mV)

## Troubleshooting

### Common Issues

**1. No activations detected**
- Try lowering `threshold_factor` (e.g., 1.5 instead of 2.0)
- Adjust `freq_range` for your data
- Check if signal quality is sufficient

**2. Too many false positives**
- Increase `threshold_factor`
- Increase `min_duration`
- Use more aggressive DWT filtering (higher Q)

**3. Memory errors with large datasets**
- Reduce `percentage_keep` in CWT analysis
- Process segments individually instead of all at once

**4. Import errors**
- Ensure all dependencies are installed
- Check Python version (3.7+ recommended)

## Scientific Background

### EMG Signal Characteristics

- **Frequency range**: 20-500 Hz (most energy: 50-150 Hz)
- **Amplitude**: µV to mV range
- **Noise sources**: Power line (50/60 Hz), motion artifacts, electrode contact

### Gait Cycle Phases

1. **Heel Strike (HS)**: Initial contact with ground
2. **Stance Phase**: Foot flat on ground
3. **Toe Off (TO)**: Foot leaves ground  
4. **Swing Phase**: Foot in air

### Muscle Functions

- **Gastrocnemius Lateralis**: Plantarflexion (push-off), stance control
- **Vastus Lateralis**: Knee extension, weight acceptance

## References

1. Merletti, R., & Parker, P. A. (2004). *Electromyography: Physiology, Engineering, and Non-Invasive Applications*. Wiley-IEEE Press.

2. Hodges, P. W., & Bui, B. H. (1996). A comparison of computer-based methods for the determination of onset of muscle contraction using electromyography. *Electroencephalography and Clinical Neurophysiology*, 101(6), 511-519.

3. Li, X., Zhou, P., & Aruin, A. S. (2007). Teager-Kaiser energy operation of surface EMG improves muscle activity onset detection. *Annals of Biomedical Engineering*, 35(9), 1532-1538.
