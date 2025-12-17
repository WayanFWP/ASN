from Utils import *
from preproceed import *
from Plot import *
from Analyzer import Analyzer
import matplotlib.pyplot as plt
import numpy as np

preprocess = dataLoader("B0102T")
data_analyzer = Analyzer(fs=preprocess.fs)
data, labels = preprocess.X, preprocess.y
print(f"Data shape: {data.shape}")  # (n_trials, n_channels, n_samples)

data_analyzer.run(data)
plot2Signal(data_analyzer.erd_data, data_analyzer.ers_data, fs=preprocess.fs, label1='ERD Band (8-11 Hz)', label2='ERS Band (26-30 Hz)')


data_analyzer.baselineNormalization(start=0, end=int(preprocess.fs * 1.0)) 
data_analyzer.motorImagery(start=int(preprocess.fs * 1.0), end=int(preprocess.fs * 5.0))

