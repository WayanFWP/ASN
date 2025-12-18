import numpy as np
from Utils import *
from Plot import *

class Analyzer:
    def __init__(self, fs=250):
        self.fs = fs
        self.alpha_power    = None
        self.beta_power     = None
        self.data           = None
        
        self.erd_bandpassed = None
        self.ers_bandpassed = None
        self.erd_squared    = None
        self.ers_squared    = None
        self.erd_movingavg  = None
        self.ers_movingavg  = None
                
        self.alpha_percent  = None
        self.beta_percent   = None

        self.refrence_free_alpha = None
        self.refrence_free_beta  = None
        
        self.window_size = 100
        
    def compute_ERP(self, data):
    # average across trials → phase-locked component
        return np.mean(data, axis=0, keepdims=True)
    
    def remove_ERP(self, data):
        erp = self.compute_ERP(data)
        return data - erp
                                                    
    def run(self, data, remove_erp=True):
        if remove_erp:
            print("Removing ERP (phase-locked activity)")
            data = self.remove_ERP(data)
        self.data = data
        self.alpha_power = self.Bandpass(data, 8, 11, band="alpha")
        self.beta_power  = self.Bandpass(data, 26, 30, band="beta")
            
    def Bandpass(self, data, Lowcut, Highcut, band="alpha"):
        bandpassed = np.zeros_like(data)
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                bandpassed[t, c, :] = BPF(
                    data[t, c, :],
                    lowcut=Lowcut,
                    highcut=Highcut,
                    fs=self.fs
                )
                
        if band == "alpha":
            self.erd_bandpassed = bandpassed.copy()
        else:
            self.ers_bandpassed = bandpassed.copy()
        return self.Squaring(bandpassed, band)

    def Squaring(self, data, band):
        squared = np.zeros_like(data)
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                squared[t, c, :] = squaring(data[t, c, :])

        if band == "alpha":
            self.erd_squared = squared.copy()
        else:
            self.ers_squared = squared.copy()
        return self.MovingAverage(squared, band, N=self.window_size)

    def MovingAverage(self, data, band, N=100):
        averaged = np.zeros_like(data)
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                averaged[t, c, :] = avaragingOverN(data[t, c, :], N)
                
        if band == "alpha":
            self.erd_movingavg = averaged.copy()
        else:            
            self.ers_movingavg = averaged.copy()
        return averaged
    
    def apply_baseline(self, baseline_start, baseline_end):
        self.alpha_percent = self.ERD_ERS(
            self.alpha_power, baseline_start, baseline_end
        )
        self.beta_percent = self.ERD_ERS(
            self.beta_power, baseline_start, baseline_end
        )

    
    # using the period before the cue as reference
    # equation: (A - R) / R * 100%
    def ERD_ERS(self, power, baseline_start, baseline_end):
        R = np.mean(power[:, :, baseline_start:baseline_end],
                    axis=2, keepdims=True)
        return (power - R) / R * 100
                    
    def motorImagery(self, start, end):
        erd = self.alpha
        ers = self.beta

        erd_mean = np.mean(erd[:, :, start:end], axis=2)
        ers_mean = np.mean(ers[:, :, start:end], axis=2)

        erd_diff = erd_mean[:, 0] - erd_mean[:, 2]  # C3 - C4
        ers_diff = ers_mean[:, 0] - ers_mean[:, 2]

        self.features = np.column_stack([
            erd_diff,
            ers_diff
        ])

        print("MI Features shape:", self.features.shape)
        
    def compute_common_average_reference(self, data):
        return data - np.mean(data, axis=1, keepdims=True)
    
    def compute_laplacian(self, data, channel_names=["EEG:C3", "EEG:Cz", "EEG:C4"]):
        laplacian = np.zeros_like(data)
        n_trials, n_channels, n_times = data.shape
        
        for t in range(n_trials):
            # C3 (left): Laplacian ≈ C3 - average(Cz)
            laplacian[t, 0, :] = data[t, 0, :] - data[t, 1, :]
            
            # Cz (center): Laplacian ≈ Cz - average(C3, C4)
            laplacian[t, 1, :] = data[t, 1, :] - 0.5 * (data[t, 0, :] + data[t, 2, :])
            
            # C4 (right): Laplacian ≈ C4 - average(Cz)
            laplacian[t, 2, :] = data[t, 2, :] - data[t, 1, :]
        
        return laplacian
    
    def apply_spatial_filter(self, method='laplacian'):
        if method == 'car':
            self.alpha = self.compute_common_average_reference(
                self.alpha_percent
            )
            self.beta = self.compute_common_average_reference(
                self.beta_percent
            )
            print("Applied Common Average Reference (CAR)")
            
        elif method == 'laplacian':
            self.alpha = self.compute_laplacian(self.alpha_percent)
            self.beta = self.compute_laplacian(self.beta_percent)
            print("Applied Surface Laplacian")
        
        else:
            raise ValueError("Method must be 'car' or 'laplacian'")