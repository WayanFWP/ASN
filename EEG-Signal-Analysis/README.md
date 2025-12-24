# EEG Motor Imagery Classification

A comprehensive Brain-Computer Interface (BCI) project for classifying motor imagery tasks (Left Hand vs Right Hand) using EEG signals. This project implements advanced signal processing techniques including ERD/ERS analysis, Common Spatial Pattern (CSP) feature extraction, and multiple machine learning classifiers with explainable AI features.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project analyzes EEG data from motor imagery tasks where subjects imagine moving either their left or right hand. The system extracts meaningful features from brain signals and uses machine learning to classify the intended movement.

## Features

### Signal Processing
- **Bandpass Filtering**: Alpha (8-11 Hz) and Beta (26-30 Hz) band extraction
- **Spatial Filtering**: Laplacian and Common Average Reference (CAR)
- **ERD/ERS Analysis**: Event-Related Desynchronization/Synchronization computation
- **CSP**: Common Spatial Pattern for optimal spatial feature extraction

### Machine Learning
- **Multiple Classifiers**:
  - Linear Discriminant Analysis (LDA)
  - Support Vector Machine (SVM)
  - Random Forest
- **Feature Engineering**: Combined CSP + ERP features
- **Explainable AI**: Feature importance analysis and per-trial explanations

### Interactive Dashboard
- **Streamlit Web Interface** with multiple analysis pipelines:
  - ERP Pipeline: ERD/ERS visualization and analysis
  - CSP Pipeline: Spatial pattern analysis and feature extraction
  - ML Pipeline: End-to-end classification with model performance metrics

## Project Structure

```
Assigment4/
├── data/
│   ├── raw/                    # Raw EEG data files (.gdf format)
├── pages/                      # Streamlit page modules
│   ├── CSPPipeline.py         # CSP feature extraction interface
│   ├── ERPPipeline.py         # ERD/ERS analysis interface
│   └── MLPipeline.py          # ML classification interface
├── Analyzer.py                 # ERD/ERS analysis and feature extraction
├── CSP.py                      # Common Spatial Pattern implementation
├── Plot.py                     # Visualization utilities
├── Utils.py                    # Signal processing utilities
├── preproceed.py              # Data loading and preprocessing
├── Streamlit.py               # Main Streamlit application
├── main.py                    # Basic pipeline example
├── main_LDA.py                # LDA classification pipeline
├── main_SVM.py                # SVM classification pipeline
├── main_RF.py                 # Random Forest classification pipeline
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or download the files):
```bash
cd "ASN\Assigment4"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
.\venv\Scripts\Activate
```

3. **Install dependencies**:
```bash
pip install numpy scipy scikit-learn matplotlib pandas streamlit mne
```

### Required Libraries
- `numpy`: Numerical computing
- `scipy`: Scientific computing and signal processing
- `scikit-learn`: Machine learning algorithms
- `matplotlib`: Data visualization
- `pandas`: Data manipulation
- `streamlit`: Interactive web interface
- `mne`: EEG data processing

## Usage

### Interactive Dashboard (Recommended)

Launch the Streamlit web interface for interactive analysis:

```bash
streamlit run Streamlit.py
```

**Features:**
- **ERD/ERS Pipeline**: Visualize brain activity patterns during motor imagery
- **CSP Pipeline**: Analyze spatial patterns and feature extraction
- **ML Pipeline**: Train classifiers and evaluate performance with real-time inference

### Command Line Scripts

#### 1. Train and Evaluate with LDA:
```bash
python main_LDA.py
```

#### 2. Train and Evaluate with SVM:
```bash
python main_SVM.py
```

#### 3. Train and Evaluate with Random Forest:
```bash
python main_RF.py
```

#### 4. Basic Pipeline Example:
```bash
python main.py
```

### Customization

**Modify training subjects** in any main script:
```python
train_subjects = ["B0101T", "B0102T", "B0201T", "B0202T"]
test_subjects = ["B0103T", "B0203T"]
```

**Adjust CSP components**:
```python
csp = CSP(csp_component=4)  # Number of spatial filters max to 6
```

**Change classifier parameters** (in Streamlit or main scripts):
- LDA: No parameters needed
- SVM: Adjust `C` and `kernel` type
- Random Forest: Modify `n_estimators`

## Methodology

### 1. Data Preprocessing
- Load EEG data from .gdf files (BCI Competition format)
- Select motor cortex channels: C3, Cz, C4
- Epoch extraction: [-1s to 3s] relative to cue onset
- Baseline period: [0s to 1s] (pre-cue)

### 2. Feature Extraction

#### A. ERD/ERS Features
1. **Bandpass Filtering**: Extract alpha (8-11 Hz) and beta (26-30 Hz) bands
2. **Power Estimation**: Square filtered signals
3. **Smoothing**: Apply moving average filter
4. **Baseline Normalization**: Compute percentage change from baseline
5. **Spatial Filtering**: Apply Laplacian or CAR filter
6. **Feature Computation**: Extract statistical features (mean, variance, etc.)

#### B. CSP Features
1. **Bandpass Filter**: 8-30 Hz for motor imagery
2. **Covariance Matrices**: Compute for each class
3. **Joint Diagonalization**: Find optimal spatial filters
4. **Variance Features**: Log-variance of filtered signals

### 3. Classification
- **Feature Combination**: Concatenate CSP + ERP features
- **Normalization**: StandardScaler for zero mean and unit variance
- **Training**: Fit classifier (LDA/SVM/Random Forest)
- **Evaluation**: Test set accuracy, confusion matrix, classification report

### 4. Explainable AI
- **Feature Importance**: Rank features by discriminative power
- **Per-Trial Analysis**: Show feature contributions for individual predictions
- **Confidence Scores**: Probability estimates for each prediction

## Results

The system achieves competitive performance on BCI Competition data:

- **Accuracy**: Typically 60-70% on test data (subject-dependent)
- **Best Features**: CSP components combined with ERD/ERS spatial contrasts

**Key Insights:**
- Alpha (8-11 Hz) desynchronization over contralateral motor cortex
- Beta (26-30 Hz) synchronization patterns
- Strong spatial discrimination between C3 and C4 channels

## Dataset

This project uses BCI Competition data with the following structure:

- **Training Files**: B01-B09 subjects, T suffix (e.g., B0101T.gdf)
- **Evaluation Files**: 3T suffix (e.g., B0103T.gdf)
- **Channels**: C3, Cz, C4 (motor cortex)
- **Sampling Rate**: 250 Hz
- **Classes**: 0 = Left Hand, 1 = Right Hand

**Note**: Ensure all data files are placed in the correct directories (`data/raw/` or `data/splited/`) before running the scripts.
