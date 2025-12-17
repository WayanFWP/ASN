import numpy as np
from Utils import *

class Analyzer:
    def __init__(self, fs=250):
        self.fs = fs
        self.erd_data = None
        self.ers_data = None
        self.data = None
        
        self.ers_percent = None
        self.erd_percent = None
        
        self.fetures = None
        self.csp_filters = None
    
    def run(self, data):
        self.data = data
        self.erd_data = self.Bandpass(data, Lowcut=8, Highcut=11)
        self.ers_data = self.Bandpass(data, Lowcut=26, Highcut=30)
        
        
    def Bandpass(self, data, Lowcut, Highcut):
        bandpassed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                bandpassed_data[i, j, :] = BPF(data[i, j, :], lowcut=Lowcut, highcut=Highcut, fs=self.fs)
        print("BandPass Filter Applied")
        return self.Squaring(bandpassed_data)
    
    def Squaring(self, data):
        squared_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                squared_data[i, j, :] = squaring(data[i, j, :])
        print("Squaring Applied")
        return self.MovingAverage(squared_data)
    
    def MovingAverage(self, data, N=10):
        averaged_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                averaged_data[i, j, :] = avaragingOverN(data[i, j, :], N=N)
        print("Moving Average Applied")
        return averaged_data
    
    def baselineNormalization(self, start, end):
        erd_baseline = np.mean(self.erd_data[:, :, start:end], axis=2, keepdims=True)
        ers_baseline = np.mean(self.ers_data[:, :, start:end], axis=2, keepdims=True)
        
        self.erd_percent = ((self.erd_data - erd_baseline) / erd_baseline) * 100
        self.ers_percent = ((self.ers_data - ers_baseline) / ers_baseline) * 100
        
    def motorImagery(self, start, end):
        erd_mean = np.mean(self.erd_percent[:, :, start:end], axis=2)
        erd_var = np.var(self.erd_percent[:, :, start:end], axis=2)
        erd_std = np.std(self.erd_percent[:, :, start:end], axis=2)
        
        ers_mean = np.mean(self.ers_percent[:, :, start:end], axis=2)
        ers_var = np.var(self.ers_percent[:, :, start:end], axis=2)
        ers_std = np.std(self.ers_percent[:, :, start:end], axis=2)
        
        self.features = np.concatenate([
            erd_mean, erd_var, erd_std,
            ers_mean, ers_var, ers_std
        ], axis=1)
        
        print(f"Temporal Features shape: {self.features.shape}")
