# Non-stationary Signal Analysis (ASN) Projects

> **Educational Project** - Semester 5 Academic Portfolio

A comprehensive collection of biomedical signal processing and analysis projects developed for educational purposes. This repository contains four major projects focused on different physiological signals: EEG, EMG, PPG, and PCG/ECG, implementing state-of-the-art signal processing techniques and machine learning algorithms.

**Learning Objectives**: Master practical signal processing, time-frequency analysis, machine learning for biosignals, and scientific computing optimization.

## Overview

This repository serves as an educational portfolio demonstrating advanced signal processing techniques applied to various biomedical signals. Each project is designed to teach fundamental concepts through hands-on implementation:

- **Time-Frequency Analysis**: CWT, STFT, DWT - Learn multi-scale signal decomposition
- **Machine Learning**: LDA, SVM, Random Forest - Understand classification pipelines
- **Feature Extraction**: CSP, ERD/ERS, HRV metrics - Master domain-specific features
- **Real-time Processing**: Streamlit dashboards - Build interactive analysis tools
- **Performance Optimization**: C-accelerated computing - Optimize computational efficiency

Each project is self-contained with its own documentation, dependencies, and data processing pipelines, making it easy to learn and experiment with individual techniques.

## Projects

### 1. CWT-STFT Analysis for PCG and ECG Signals

**Location**: `CWT-STFT_analysis/`

Analyzes Phonocardiogram (PCG) and Electrocardiogram (ECG) signals to detect and characterize heart sounds (S1 and S2).


**Quick Start**:
```bash
cd CWT-STFT_analysis
python main.py
```

[**→ Full Documentation**](CWT-STFT_analysis/README.md)

---

### 2. EEG Motor Imagery Classification

**Location**: `EEG-Signal-Analysis/`

Brain-Computer Interface (BCI) system for classifying motor imagery tasks (Left Hand vs Right Hand movement) using EEG signals.

**Quick Start**:
```bash
cd EEG-Signal-Analysis
streamlit run Streamlit.py
```

[**→ Full Documentation**](EEG-Signal-Analysis/README.md)

---

### 3. Movement Signal Analysis (EMG)

**Location**: `Movement-Signal-Analysis/`

EMG signal processing toolkit for gait cycle detection and muscle activation analysis during walking.

**Analyzed Muscles**:
- Gastrocnemius Lateralis (GL) - calf muscle
- Vastus Lateralis (VL) - thigh muscle

**Quick Start**:
```bash
cd Movement-Signal-Analysis
python main.py
```

[**→ Full Documentation**](Movement-Signal-Analysis/README.md)

---

### 4. PPG Multiscale Decomposer

**Location**: `ppg-multiscale-decomposer/`

Comprehensive Photoplethysmography (PPG) signal analyzer using multi-scale decomposition for extracting cardiovascular and respiratory parameters.

**Metrics Extracted**:
- **Time-domain HRV**: SDNN, RMSSD, pNN50, TINN
- **Frequency-domain HRV**: VLF, LF, HF power, LF/HF ratio
- **Nonlinear**: Poincaré plot analysis (SD1, SD2)

**Quick Start**:
```bash
cd ppg-multiscale-decomposer
streamlit run streamlit_app.py
```

[**→ Full Documentation**](ppg-multiscale-decomposer/README.md)

---

## Tech Stack

### Development Tools
- **Languages**: Python 3.8+, C (for acceleration)
- **Core Libraries**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Plotly, Seaborn
- **ML**: Scikit-learn, MNE-Python
- **UI**: Streamlit
- **Data Formats**: CSV, GDF, WFDB

### Performance Optimization
- **C Extensions**: FFT and convolution acceleration
- **Vectorization**: NumPy array operations
- **Memory Management**: Efficient array handling
