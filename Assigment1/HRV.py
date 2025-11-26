import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
from Utils import *  

class TimeDomain:
    """Container and calculator for time-domain HRV features."""
    def __init__(self, rr_intervals):
        self.rr_intervals = np.asarray(rr_intervals, dtype=float)
        self.n = len(self.rr_intervals)
        self.mean_rr = np.mean(self.rr_intervals) if self.n > 0 else np.nan
        
        
    def _rr_ms(self):
        """Return RR intervals in milliseconds."""
        rr = np.array(self.rr_intervals, dtype=float) * 1000.0
        # rr = rr[(rr >= 300) & (rr <= 2000)] # Filter to valid range
        return rr

    def SDANN(self):
        """Standard Deviation of the Average NN intervals.
        using formula sqrt( mean( (RRi - meanRR)^2 ) )
        """
        rr_ms = self._rr_ms()
        sdann = float(np.sqrt(1/self.n * np.sum((rr_ms - self.mean_rr) ** 2)))
        return sdann 
         
    def SDNN(self):
        """
        Standard Deviation of NN intervals.
        using formula sqrt( mean( (RRi - meanRR)^2 ) )
        """
        rr_ms = self._rr_ms()
        sdnn = float(np.std(rr_ms, ddof=1)) 
        return sdnn
    
    def RMSSD(self):
        """
        Root Mean Square of Successive Differences.
        Using formula sqrt( mean( (RRi+1 - RRi)^2 ) )
        """
        rr_array = self._rr_ms()
        diff = np.diff(rr_array)
        rmssd = float(np.sqrt(np.mean(diff**2))) 
        return rmssd
    
    def NN50(self):
        if self.n < 2:
            return 0
        diff_rr = np.abs(np.diff(self.rr_intervals))
        return np.sum(diff_rr > 0.05)  # 50 ms = 0.05 s

    def pNN50(self):
        rr_array = self._rr_ms()
        diff = np.diff(rr_array)
        pnn50 = np.sum(np.abs(diff) > 50.0) / len(diff) * 100.0  # 50ms threshold
        return pnn50

    def SDSD(self):
        rr_ms = self._rr_ms()
        diffs = np.diff(rr_ms)
        return float(np.std(diffs, ddof=1))

    def HTI(self, bin_width_ms=7.8125):
        rr_ms = self._rr_ms()
        if rr_ms.size == 0:
            return np.nan
        bins = np.arange(rr_ms.min(), rr_ms.max() + bin_width_ms, bin_width_ms)
        if bins.size < 2:
            bins = np.array([rr_ms.min(), rr_ms.max() + bin_width_ms])
        hist, _ = np.histogram(rr_ms, bins=bins)
        max_count = hist.max() if hist.size > 0 else 0
        return float(rr_ms.size) / (float(max_count) + 1e-12)

    def TINN(self, bin_width_ms=7.8125, plot=False):
        nn_intervals = np.asarray(self._rr_ms())  # Convert to ms

        nn_intervals = nn_intervals[(nn_intervals >= 300) & (nn_intervals <= 2000)]
        if nn_intervals.size < 3:
            print("Not enough valid NN intervals for TINN calculation.")
            return np.nan

        min_t, max_t = np.min(nn_intervals), np.max(nn_intervals)
        bins = np.arange(min_t, max_t + bin_width_ms, bin_width_ms)
        hist, edges = np.histogram(nn_intervals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        k = int(np.argmax(hist))
        X, Y = centers[k], hist[k]
        total_counts = hist.sum()

        if Y <= 0:
            tinn_area = np.nan
        else:
            tinn_area = 2.0 * total_counts / float(Y) * float(bin_width_ms)

        nonzero_idx = np.nonzero(hist)[0]
        if nonzero_idx.size >= 2:
            tinn_constrained = centers[nonzero_idx[-1]] - centers[nonzero_idx[0]]
        else:
            tinn_constrained = np.nan

        # Debug info
        # print(f"Min RR: {min_t:.1f} ms, Max RR: {max_t:.1f} ms")
        # print(f"Range = {max_t - min_t:.1f} ms, Mode = {X:.1f} ms, Peak = {Y}")
        # print(f"TINN (area-based) = {tinn_area:.1f} ms,  TINN (constrained) = {tinn_constrained:.1f} ms")

        return tinn_area
    
    def CVNN(self):
        """Coefficient of Variation of NN (%) = (SDNN / meanNN) * 100."""
        rr_ms = self._rr_ms()
        if rr_ms.size < 2:
            return np.nan
        sdnn = np.std(rr_ms, ddof=1)
        mean_nn = np.mean(rr_ms)
        return float((sdnn / (mean_nn + 1e-12)) * 100.0)

    def CVSD(self):
        """Coefficient of Variation of Successive Differences (%) = (RMSSD / meanNN) * 100."""
        rr_ms = self._rr_ms()
        if rr_ms.size < 2:
            return np.nan
        diffs = np.diff(rr_ms)
        if diffs.size == 0:
            return np.nan
        rmssd = np.sqrt(np.mean(diffs**2))
        mean_nn = np.mean(rr_ms)
        return float((rmssd / (mean_nn + 1e-12)) * 100.0)
    
    def Skewness(self, unbiased=True):
        """
        Calculate skewness of RR intervals.
        
        Parameters:
        - unbiased: Apply bias correction (default True)
        - method: 'standard' or 'diff' - use standard skewness or difference-based skewness
        """
        rr_ms = self._rr_ms()
        # Alternative method using successive differences
        if rr_ms.size < 2:
            return 0.0
            
        diff = np.diff(rr_ms)
        if diff.size == 0:
            return 0.0
            
        numerator = np.mean(diff**3)
        denominator = (np.sqrt(np.mean(diff**2)))**3
        skewness = numerator / denominator if denominator != 0 else 0
        return float(skewness)
        
    def get_features(self):
        return {
            "SDNN_ms": self.SDNN(),
            "SDANN_ms": self.SDANN(),
            "RMSSD_ms": self.RMSSD(),
            "NN50_count": self.NN50(),
            "pNN50_percent": self.pNN50(),
            "SDSD_ms": self.SDSD(),
            "HTI": self.HTI(),
            "TINN_ms": self.TINN(),
            "CVNN_percent": self.CVNN(),
            "CVSD_percent": self.CVSD(),
            "Skewness": self.Skewness()
        }


class FrequencyDomain:
    """Container and calculator for frequency-domain HRV features."""
    def __init__(self, rr_intervals, interp_freq=4.0):
        self.rr_intervals = np.asarray(rr_intervals, dtype=float)
        self.n = len(self.rr_intervals)
        self.fs_interp = interp_freq

        # Frequency bands
        self.ulf_band = (0.0001, 0.003)
        self.vlf_band = (0.003, 0.04)
        self.lf_band = (0.04, 0.15)
        self.hf_band = (0.15, 0.4)

        self.freqs = None
        self.psd = None

    def compute_psd(self):
        if self.n < 2:
            return np.array([]), np.array([])
        fs = 1.0 / np.mean(self.rr_intervals)
        freq, psd = welch(self.rr_intervals, fs=fs, nperseg=min(256, self.n))
        self.freqs, self.psd = freq, psd
        return freq, psd

    def calculate_freq_domain_features(self, fxx, pxx):
        """Calculates VLF, LF, HF power, LF/HF ratio, LFnu, HFnu, and peak frequencies."""
        features = {}

        # --- Define frequency bands ---
        vlf_band = (fxx >= 0.003) & (fxx < 0.04)
        lf_band = (fxx >= 0.04) & (fxx < 0.15)
        hf_band = (fxx >= 0.15) & (fxx < 0.4)

        # --- Integrate PSD to get power in each band ---
        features["VLF"] = trapezoid(pxx[vlf_band], fxx[vlf_band]) if np.any(vlf_band) else 0
        features["LF"] = trapezoid(pxx[lf_band], fxx[lf_band]) if np.any(lf_band) else 0
        features["HF"] = trapezoid(pxx[hf_band], fxx[hf_band]) if np.any(hf_band) else 0

        # --- Calculate derived ratios ---
        lf_plus_hf = features["LF"] + features["HF"]
        features["Total Power (LF+HF)"] = lf_plus_hf
        features["LF/HF Ratio"] = features["LF"] / features["HF"] if features["HF"] > 0 else 0
        features["LFnu"] = (features["LF"] / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
        features["HFnu"] = (features["HF"] / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0

        # --- Find Peak Frequency in LF and HF bands ---
        if np.any(lf_band) and np.any(pxx[lf_band]):
            lf_peak_idx = np.argmax(pxx[lf_band])
            features["LF Peak Freq (Hz)"] = fxx[lf_band][lf_peak_idx]
        else:
            features["LF Peak Freq (Hz)"] = 0

        if np.any(hf_band) and np.any(pxx[hf_band]):
            hf_peak_idx = np.argmax(pxx[hf_band])
            features["HF Peak Freq (Hz)"] = fxx[hf_band][hf_peak_idx]
        else:
            features["HF Peak Freq (Hz)"] = 0

        return features

    def get_features(self):
        """Compute and return all frequency domain features."""
        freq, psd = self.compute_psd()
        if freq.size == 0 or psd.size == 0:
            return {}
        return self.calculate_freq_domain_features(freq, psd)

class NonLinearDomain:
    """Container for nonlinear HRV features (entropy, Poincaré, etc.)."""
    def __init__(self, rr_intervals):
        self.rr_intervals = np.asarray(rr_intervals, dtype=float)
        self.n = len(self.rr_intervals)
        self.timeDom = TimeDomain(self.rr_intervals)
        
    def _rr_ms(self):
        """Return RR intervals in milliseconds."""
        return np.array(self.rr_intervals, dtype=float) * 1000.0
        
    def compute_poincare(self):
        """
        Poincaré plot indices SD1 and SD2 (Brennan et al. 2001).
        SD1: standard deviation perpendicular to line of identity (short-term variability)
        SD2: standard deviation along line of identity (long-term variability)
        SD1/SD2 ratio
        """
        rr_ms = self._rr_ms()
        if rr_ms.size < 2:
            return {"SD1_ms": np.nan, "SD2_ms": np.nan, "SD1_SD2_ratio": np.nan}
        
        # RR(n) vs RR(n+1)
        rr_n = rr_ms[:-1]
        rr_n1 = rr_ms[1:]
        
        diff = rr_n1 - rr_n
        sum_rr = rr_n1 + rr_n        
        
        """
        SD1 = sqrt(0.5) * RMSSD
        SD2 = sqrt(2 * SDNN**2 - 0.5 * RMSSD**2)
        """
        sd1 = np.sqrt(0.5) * self.timeDom.RMSSD()
        sd2 = np.sqrt(2*(self.timeDom.SDNN()**2) - 0.5 * (self.timeDom.RMSSD()**2))
        
        sd_ratio = sd1 / (sd2 + 1e-12)
        
        return {
            "SD1_ms": float(sd1),
            "SD2_ms": float(sd2),
            "SD1_SD2_ratio": float(sd_ratio)
        }
    
    def compute_all(self):
        """Compute all nonlinear features and return as dictionary."""
        poincare = self.compute_poincare()
        
        return {**poincare}

class HRV:
    """Main HRV handler combining Time, Frequency, and Nonlinear domains."""
    def __init__(self, rr_intervals):
        self.time = TimeDomain(rr_intervals)
        self.freq = FrequencyDomain(rr_intervals)
        self.non_linear = NonLinearDomain(rr_intervals)
        # self.classifier = HRVClassifier()

    def compute_all(self):
        return {
            "time_domain": self.time.get_features(),
            "frequency_domain": self.freq.get_features(),
            "nonlinear_domain": self.non_linear.compute_all()
        }

    def print_time_features(self):
        feats = self.time.get_features()
        print("\n=== HRV TimeDomain Features ===")
        for key, val in feats.items():
            print(f"{key}: {val:.4f}" if isinstance(val, (float, np.floating)) else f"{key}: {val}")
    
    def print_frequency_features(self):
            features = self.freq.get_features()
            print("\n=== HRV Frequency Domain Features ===")
            if features:
                for key, value in features.items():
                    print(f"{key}: {value:.4f}" if not np.isnan(value) else f"{key}: N/A")
            else:
                print("Not enough RR intervals to compute frequency domain features.")
                        
    def print_nonlinear_features(self):
        features = self.non_linear.compute_all()
        print("\n=== HRV Nonlinear Features ===")
        for key, value in features.items():
            print(f"{key}: {value:.4f}" if not np.isnan(value) else f"{key}: N/A")