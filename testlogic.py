from Assigment1.Coeficient import Coeficient
from Assigment1.Utils import *
from Assigment1.SpectrumAnalyzer import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize Coeficient
fs = 125
factor = 45
fs = fs / factor

coef = Coeficient(fs)
coef.initialize_qj_filter()

load_file = pd.read_csv("data/bidmc_01_Signals.csv")

# Load data
print("Available columns:", load_file.columns.tolist())
selected_signal = "PLETH" if "PLETH" in load_file.columns else load_file.columns[2]
time = load_file.iloc[:, 0].values  # Use first column for time
signal = load_file[selected_signal].values
data = pd.DataFrame({'Time': time, 'Signal': signal})

mean_signal = np.mean(data['Signal'].values)
data['Signal'] = data['Signal'].values - mean_signal  # Centering
# Apply DWT1-8
selected_j = 7
signal_DWT = coef.applying(data['Signal'].values, specific_j=selected_j, factor=factor)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(data['Signal'], label="Raw Data")
plt.ylabel("Amp")



# plotFreqSpectrum(signal_DWT[selected_j], fs, len(signal_DWT[selected_j]))
# plotTimeDomain(signal_DWT[selected_j])

plt.show()
