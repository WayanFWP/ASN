import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

namefile = "a0042"
args = "dat/raw/"+namefile
data = wfdb.rdrecord(args)

print(data.fs)
print(data.sig_name)
print(data.units)

df = pd.DataFrame(data.p_signal, columns=data.sig_name)

PCG_data = df.iloc[:, 0]
ECG_data = df.iloc[:, 1]

time = np.arange(len(df)) / data.fs

df.insert(0, 'Time', time)
df.set_index('Time', inplace=True)
print(df.head())
print(PCG_data.head(), ECG_data.head())

# Comment this if you don't want to plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, ECG_data, label='ECG')
plt.plot(df.index, PCG_data, label='PCG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('ECG and PCG Signals')
plt.legend()
plt.show()

# Save to CSV
df.to_csv(f'dat/{namefile}.csv')