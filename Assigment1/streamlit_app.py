import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import welch, hilbert
from scipy.interpolate import interp1d
import io
import json
from contextlib import redirect_stdout, redirect_stderr

# Import your modules
from Coeficient import Coeficient
from Utils import *
from HRV import HRV as feature
from Analyze import Respiratory, Vasometric, HeartRate

def main():
    st.set_page_config(
        page_title="Signal Analysis Pipeline",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Signal Analysis Pipeline")
    st.markdown("---")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Select your signal data CSV file"
        )
        
        # Parameters section
        st.subheader("Analysis Parameters")
        
        # Sampling frequency
        fs_original = st.number_input(
            "Original Sampling Frequency (Hz)", 
            min_value=1.0, 
            max_value=1000.0, 
            value=50.0, 
            step=1.0
        )
        
        # Downsampling factor
        factor = st.number_input(
            "Downsampling Factor", 
            min_value=1.0, 
            max_value=10.0, 
            value=1.15, 
            step=0.01
        )
        
        # Effective sampling frequency (calculated)
        fs_effective = fs_original / factor
        st.info(f"Effective FS: {fs_effective:.2f} Hz")
        
        # Filter parameters
        st.subheader("Filter Parameters")
        col1, col2 = st.columns(2)
        with col1:
            bp_low = st.number_input("BP Low (Hz)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
        with col2:
            bp_high = st.number_input("BP High (Hz)", min_value=1.0, max_value=100.0, value=45.0, step=1.0)
        
        # DWT levels
        st.subheader("DWT Levels")
        col1, col2 = st.columns(2)
        with col1:
            j_resp = st.number_input("Respiratory Level", min_value=1, max_value=10, value=7, step=1)
        with col2:
            j_vaso = st.number_input("Vasometric Level", min_value=1, max_value=10, value=8, step=1)
    
    # Main content area
    if uploaded_file is not None:
        # Load and display file info
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
            # Column selection
            st.subheader("📋 Data Preview")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("**Available Columns:**")
                selected_column = st.selectbox(
                    "Select Signal Column", 
                    df.columns.tolist(),
                    index=1 if len(df.columns) > 1 else 0
                )
                
                # Show basic stats
                if selected_column:
                    signal_stats = df[selected_column].describe()
                    st.write("**Signal Statistics:**")
                    st.dataframe(signal_stats, use_container_width=True)
            
            # Auto-run analysis when file is loaded and column is selected
            if selected_column:
                run_analysis(df, selected_column, fs_effective, factor, bp_low, bp_high, j_resp, j_vaso)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example of expected data format
        st.subheader("📝 Expected Data Format")
        example_data = pd.DataFrame({
            'Time': [0, 0.02, 0.04, 0.06, 0.08],
            'Signal': [0.1, 0.2, -0.1, 0.3, 0.15],
            'Other_Column': [1, 2, 3, 4, 5]
        })
        st.dataframe(example_data, use_container_width=True)
        st.caption("Your CSV should have time-series signal data in columns")

@st.cache_data
def run_analysis_cached(signal_data, fs_effective, factor, bp_low, bp_high, j_resp, j_vaso):
    """Cached analysis function to prevent re-computation"""
    
    # Initialize components
    coef = Coeficient(fs_effective)
    coef.gui_mode = True  # Disable popup plots
    coef.initialize_qj_filter()
    
    HR = HeartRate(fs_effective)
    Resp = Respiratory(fs_effective)
    Vaso = Vasometric(fs_effective)
    
    # Load and preprocess signal
    signal = downSample(signal_data, factor)
    
    # Preprocessing
    mean_signal = np.mean(signal)
    signal = signal - mean_signal  # Centering
    signal = BPF(signal, bp_low, bp_high, fs_effective)
    
    # Heart Rate Analysis
    signal_Hr, peaks_Hr, BPM = HR.analyze(signal)
    rr_intervals = np.diff(peaks_Hr / fs_effective)
    
    # Respiratory Analysis
    signal_DWT = coef.applying(signal, specific_j=j_resp)
    resp_data, resp_peaks, BrPM = Resp.analyze(signal_DWT[j_resp])
    freq, magnitude, peak_freq, peak_mag = Resp.get_freq()
    
    # Vasometric Analysis
    signal_dwtvaso = coef.applying(signal, specific_j=j_vaso)
    vaso_freq, vaso_mag, peak_vaso, vaso_peak_mag = Vaso.analyze(signal_dwtvaso[j_vaso])
    
    # HRV Analysis
    if len(rr_intervals) > 1:
        hrv = feature(rr_intervals)
        time_features = hrv.time.get_features()
        freq_features = hrv.freq.get_features()
        nonlinear_features = hrv.non_linear.compute_all()
    else:
        time_features = {}
        freq_features = {}
        nonlinear_features = {}
    
    return {
        'signal_Hr': signal_Hr,
        'peaks_Hr': peaks_Hr,
        'BPM': BPM,
        'rr_intervals': rr_intervals,
        'resp_signal': signal_DWT[j_resp],
        'resp_peaks': resp_peaks,
        'BrPM': BrPM,
        'peak_freq': peak_freq,
        'resp_freq': freq,
        'resp_magnitude': magnitude,
        'vaso_signal': signal_dwtvaso[j_vaso],
        'peak_vaso': peak_vaso,
        'vaso_freq': vaso_freq,
        'vaso_magnitude': vaso_mag,
        'time_features': time_features,
        'freq_features': freq_features,
        'nonlinear_features': nonlinear_features,
        'original_signal': signal,
        'coef': coef
    }

def run_analysis(df, selected_column, fs_effective, factor, bp_low, bp_high, j_resp, j_vaso):
    """Run the complete signal analysis pipeline with progress indicators"""
    
    # Show progress
    with st.spinner('🔄 Running signal analysis...'):
        try:
            # Get signal data
            signal_data = df[selected_column].values
            
            # Run cached analysis
            results = run_analysis_cached(
                signal_data, fs_effective, factor, bp_low, bp_high, j_resp, j_vaso
            )
            
            # Display results
            display_results(
                results['signal_Hr'], results['peaks_Hr'], results['BPM'], results['rr_intervals'],
                results['resp_signal'], results['resp_peaks'], results['BrPM'], 
                results['peak_freq'], results['resp_freq'], results['resp_magnitude'],
                results['vaso_signal'], results['peak_vaso'], results['vaso_freq'], results['vaso_magnitude'],
                results['time_features'], results['freq_features'], results['nonlinear_features'],
                fs_effective, results['original_signal'], results['coef'], j_resp, j_vaso
            )
            
            if len(results['rr_intervals']) <= 1:
                st.warning("⚠️ Not enough RR intervals for comprehensive HRV analysis")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# ========== DATA STRUCTURES ==========
class SignalData:
    """Modular data container for analysis results"""
    def __init__(self, results, fs, j_resp, j_vaso):
        # Signal data
        self.signal_Hr = results['signal_Hr']
        self.peaks_Hr = results['peaks_Hr']
        self.BPM = results['BPM']
        self.rr_intervals = results['rr_intervals']
        
        self.resp_signal = results['resp_signal']
        self.resp_peaks = results['resp_peaks']
        self.BrPM = results['BrPM']
        self.peak_freq = results['peak_freq']
        self.resp_freq = results['resp_freq']
        self.resp_magnitude = results['resp_magnitude']
        
        self.vaso_signal = results['vaso_signal']
        self.peak_vaso = results['peak_vaso']
        self.vaso_freq = results['vaso_freq']
        self.vaso_magnitude = results['vaso_magnitude']
        
        # Features
        self.time_features = results['time_features']
        self.freq_features = results['freq_features']
        self.nonlinear_features = results['nonlinear_features']
        
        # Metadata
        self.original_signal = results['original_signal']
        self.coef = results['coef']
        self.fs = fs
        self.j_resp = j_resp
        self.j_vaso = j_vaso
        
        # Time axes
        self.time_hr = np.arange(len(self.signal_Hr)) / fs
        self.time_resp = np.arange(len(self.resp_signal)) / fs
        self.time_vaso = np.arange(len(self.vaso_signal)) / fs
        self.time_orig = np.arange(len(self.original_signal)) / fs

class PlotManager:
    """Manages synchronized plotting with shared x-axis"""
    def __init__(self, data: SignalData):
        self.data = data
        self.shared_xaxis = f"x{hash(str(data.time_hr)) % 1000}"  # Unique identifier
    
    def create_synchronized_figure(self, rows, cols, subplot_titles, specs=None):
        """Create figure with synchronized x-axes for time domain plots"""
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
            shared_xaxes=True  # Enable shared x-axis
        )
        return fig
    
    def add_time_series_trace(self, fig, x_data, y_data, name, row, col, **kwargs):
        """Add time series trace with consistent styling"""
        default_props = {
            'mode': 'lines',
            'line': dict(width=1.5),
            'showlegend': True
        }
        default_props.update(kwargs)
        
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data, name=name, **default_props),
            row=row, col=col
        )
    
    def add_peaks_trace(self, fig, x_peaks, y_peaks, name, row, col, **kwargs):
        """Add peak markers with consistent styling"""
        default_props = {
            'mode': 'markers',
            'marker': dict(size=8, symbol='circle'),
            'showlegend': True
        }
        default_props.update(kwargs)
        
        fig.add_trace(
            go.Scatter(x=x_peaks, y=y_peaks, name=name, **default_props),
            row=row, col=col
        )

class MetricsDisplay:
    """Handles metrics display and formatting"""
    @staticmethod
    def show_basic_metrics(data: SignalData):
        """Display basic analysis metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💓 Heart Rate", f"{data.BPM:.1f}", "BPM")
        
        with col2:
            st.metric("🫁 Breathing Rate", f"{data.BrPM:.1f}", "BrPM")
        
        with col3:
            st.metric("🌊 Respiratory Freq", f"{data.peak_freq:.4f}", "Hz")
        
        with col4:
            st.metric("🩸 Vasometric Freq", f"{data.peak_vaso:.4f}", "Hz")
    
    @staticmethod
    def show_signal_stats(signal, fs, title):
        """Display signal statistics in a consistent format"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(f"{title} Mean", f"{np.mean(signal):.4f}")
        with col2:
            st.metric(f"{title} Std", f"{np.std(signal):.4f}")
        with col3:
            st.metric("Length", f"{len(signal)} samples")
        with col4:
            st.metric("Duration", f"{len(signal)/fs:.2f} sec")

# ========== ANALYSIS MODULES ==========
class SignalProcessingModule:
    """Signal processing analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create signal preprocessing visualization plots"""
        st.subheader("🔧 Signal Preprocessing")
        st.write("**Shows the effect of preprocessing steps on the original signal**")
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Original Signal', 'Processed Signal (Centered + Bandpass Filtered)'],
            vertical_spacing=0.15
        )
        
        # Original signal
        fig.add_trace(
            go.Scatter(x=data.time_orig, y=data.original_signal, name='Original', 
                      line=dict(color='gray', width=1)),
            row=1, col=1
        )
        
        # Processed signal
        fig.add_trace(
            go.Scatter(x=data.time_hr, y=data.signal_Hr, name='Processed', 
                      line=dict(color='blue', width=1)),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Amplitude")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Mean", f"{np.mean(data.original_signal):.4f}")
            st.metric("Original Std", f"{np.std(data.original_signal):.4f}")
        with col2:
            st.metric("Processed Mean", f"{np.mean(data.signal_Hr):.4f}")
            st.metric("Processed Std", f"{np.std(data.signal_Hr):.4f}")
        with col3:
            st.metric("Signal Length", f"{len(data.original_signal)} samples")
            st.metric("Duration", f"{len(data.original_signal)/data.fs:.2f} seconds")

class HeartRateModule:
    """Heart rate analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create detailed heart rate analysis plots"""
        st.subheader("💓 Heart Rate Analysis")
        st.write("**Detailed analysis of heart rate detection and variability**")
        
        # Create subplots for HR analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Heart Rate Signal with Peak Detection',
                'RR Interval Tachogram',
                'RR Interval Distribution',
                'Beat-to-Beat Interval Analysis'
            ],
            specs=[[{"colspan": 2}, None],
                   [{}, {}]],
            vertical_spacing=0.15
        )
        
        # 1. HR Signal with peaks
        fig.add_trace(
            go.Scatter(x=data.time_hr, y=data.signal_Hr, name='HR Signal', 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.time_hr[data.peaks_Hr], y=data.signal_Hr[data.peaks_Hr], 
                      mode='markers', name='Detected Peaks', 
                      marker=dict(color='blue', size=8, symbol='triangle-up')),
            row=1, col=1
        )
        
        # 2. RR Interval tachogram
        if len(data.rr_intervals) > 0:
            rr_times = data.peaks_Hr[1:] / data.fs
            rr_ms = data.rr_intervals * 1000
            
            fig.add_trace(
                go.Scatter(x=rr_times, y=rr_ms, mode='lines+markers',
                          name='RR Intervals', line=dict(color='green', width=2),
                          marker=dict(size=4)),
                row=2, col=1
            )
            
            # 3. RR Interval histogram
            fig.add_trace(
                go.Histogram(x=rr_ms, nbinsx=20, name='RR Distribution',
                            marker_color='orange', opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="RR Interval (ms)", row=2, col=2)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="RR Interval (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # HR Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average BPM", f"{data.BPM:.1f}")
        with col2:
            st.metric("Peak Count", f"{len(data.peaks_Hr)}")
        with col3:
            if len(data.rr_intervals) > 0:
                st.metric("Avg RR (ms)", f"{np.mean(data.rr_intervals)*1000:.1f}")
        with col4:
            if len(data.rr_intervals) > 0:
                st.metric("RR Std (ms)", f"{np.std(data.rr_intervals)*1000:.1f}")

class RespiratoryModule:
    """Respiratory signal analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create detailed respiratory analysis plots"""
        st.subheader("🫁 Respiratory Signal Analysis")
        st.write("**Analysis of breathing patterns and frequency characteristics**")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Respiratory Signal with Peak Detection',
                'Frequency Spectrum',
                'Respiratory Rate Over Time',
                'Peak Detection Statistics'
            ],
            specs=[[{"colspan": 2}, None],
                   [{}, {}]],
            vertical_spacing=0.15
        )
        
        # 1. Respiratory signal with peaks
        fig.add_trace(
            go.Scatter(x=data.time_resp, y=data.resp_signal, name='Respiratory Signal', 
                      line=dict(color='green', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.time_resp[data.resp_peaks], y=data.resp_signal[data.resp_peaks], 
                      mode='markers', name='Detected Peaks', 
                      marker=dict(color='orange', size=8, symbol='circle')),
            row=1, col=1
        )
        
        # 2. Frequency spectrum
        fig.add_trace(
            go.Scatter(x=data.resp_freq, y=data.resp_magnitude, name='Frequency Spectrum',
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        # Mark peak frequency
        peak_idx = np.argmax(data.resp_magnitude[1:]) + 1
        fig.add_trace(
            go.Scatter(x=[data.resp_freq[peak_idx]], y=[data.resp_magnitude[peak_idx]],
                      mode='markers', name='Peak Frequency',
                      marker=dict(color='red', size=12, symbol='star')),
            row=2, col=1
        )
        
        # 3. Respiratory rate calculation over time windows
        if len(data.resp_peaks) > 2:
            window_size = min(len(data.resp_peaks)//3, 10)
            breathing_rates = []
            window_times = []
            
            for i in range(0, len(data.resp_peaks)-window_size, window_size//2):
                window_peaks = data.resp_peaks[i:i+window_size]
                window_duration = (window_peaks[-1] - window_peaks[0]) / data.fs
                if window_duration > 0:
                    rate = (len(window_peaks)-1) / window_duration * 60
                    breathing_rates.append(rate)
                    window_times.append(window_peaks[len(window_peaks)//2] / data.fs)
            
            if breathing_rates:
                fig.add_trace(
                    go.Scatter(x=window_times, y=breathing_rates, mode='lines+markers',
                              name='Breathing Rate', line=dict(color='purple', width=2)),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude", row=2, col=1)
        fig.update_yaxes(title_text="Breathing Rate (BrPM)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Respiratory Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Breathing Rate", f"{data.BrPM:.1f} BrPM")
        with col2:
            st.metric("Peak Frequency", f"{data.peak_freq:.4f} Hz")
        with col3:
            st.metric("Breath Count", f"{len(data.resp_peaks)}")
        with col4:
            duration = len(data.resp_signal) / data.fs
            st.metric("Analysis Duration", f"{duration:.1f} sec")

class VasometricModule:
    """Vasometric signal analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create vasometric signal analysis plots"""
        st.subheader("🩸 Vasometric Signal Analysis")
        st.write("**Analysis of vascular oscillations and blood flow patterns**")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Vasometric Signal',
                'Frequency Spectrum',
                'Power Spectral Density',
                'Signal Statistics'
            ],
            vertical_spacing=0.15
        )
        
        # 1. Vasometric signal
        fig.add_trace(
            go.Scatter(x=data.time_vaso, y=data.vaso_signal, name='Vasometric Signal',
                      line=dict(color='purple', width=1)),
            row=1, col=1
        )
        
        # 2. Frequency spectrum
        fig.add_trace(
            go.Scatter(x=data.vaso_freq, y=data.vaso_magnitude, name='Magnitude Spectrum',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # Mark peak frequency
        peak_idx = np.argmax(data.vaso_magnitude[1:]) + 1
        fig.add_trace(
            go.Scatter(x=[data.vaso_freq[peak_idx]], y=[data.vaso_magnitude[peak_idx]],
                      mode='markers', name='Peak Frequency',
                      marker=dict(color='red', size=12, symbol='star')),
            row=1, col=2
        )
        
        # 3. Power spectral density using Welch's method
        try:
            freqs, psd = welch(data.vaso_signal, fs=data.fs, nperseg=min(256, len(data.vaso_signal)//4))
            fig.add_trace(
                go.Scatter(x=freqs, y=psd, name='PSD (Welch)',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        except:
            st.info("PSD calculation requires longer signal")
        
        # 4. Signal envelope and characteristics
        try:
            analytic_signal = hilbert(data.vaso_signal)
            envelope = np.abs(analytic_signal)
            fig.add_trace(
                go.Scatter(x=data.time_vaso, y=envelope, name='Signal Envelope',
                          line=dict(color='orange', width=2)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=data.time_vaso, y=data.vaso_signal, name='Original',
                          line=dict(color='purple', width=1, dash='dot')),
                row=2, col=2
            )
        except:
            pass
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
        fig.update_yaxes(title_text="Amplitude")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Vasometric Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Peak Frequency", f"{data.peak_vaso:.4f} Hz")
        with col2:
            st.metric("Signal RMS", f"{np.sqrt(np.mean(data.vaso_signal**2)):.4f}")
        with col3:
            st.metric("Signal Range", f"{np.ptp(data.vaso_signal):.4f}")
        with col4:
            st.metric("Zero Crossings", f"{len(np.where(np.diff(np.sign(data.vaso_signal)))[0])}")

class HRVModule:
    """HRV analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create comprehensive HRV analysis plots"""
        st.subheader("🧠 Heart Rate Variability (HRV) Analysis")
        st.write("**Comprehensive analysis of heart rate variability patterns**")
        
        if len(data.rr_intervals) < 2:
            st.warning("⚠️ Insufficient RR intervals for HRV analysis")
            return
        
        HRVModule.create_detailed_analysis(data, plot_manager)
    
    @staticmethod
    def create_detailed_analysis(data: SignalData, plot_manager: PlotManager):
        """Create detailed HRV analysis with plots and features"""
        # Create HRV plots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'RR Interval Tachogram',
                'RR Interval Histogram', 
                'Poincaré Plot',
                'Frequency Domain Analysis',
                'Time Domain Trends',
                'Statistical Summary'
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        rr_ms = data.rr_intervals * 1000
        rr_indices = np.arange(len(rr_ms))
        
        # 1. RR Tachogram
        fig.add_trace(
            go.Scatter(x=rr_indices, y=rr_ms, mode='lines+markers',
                      name='RR Intervals', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. RR Histogram
        fig.add_trace(
            go.Histogram(x=rr_ms, nbinsx=20, name='Distribution',
                        marker_color='green', opacity=0.7),
            row=1, col=2
        )
        
        # 3. Poincaré Plot
        if len(rr_ms) > 1:
            rr1 = rr_ms[:-1]
            rr2 = rr_ms[1:]
            fig.add_trace(
                go.Scatter(x=rr1, y=rr2, mode='markers',
                          name='Poincaré', marker=dict(color='red', size=4)),
                row=1, col=3
            )
        
        # 4. Frequency Domain (if available)
        if data.freq_features:
            try:
                # Interpolate RR intervals for frequency analysis
                time_rr = np.cumsum(np.concatenate([[0], data.rr_intervals]))
                interp_freq = 4.0
                time_interp = np.arange(0, time_rr[-1], 1/interp_freq)
                
                if len(time_rr) > 1 and len(time_interp) > 10:
                    f_interp = interp1d(time_rr, np.concatenate([[rr_ms[0]], rr_ms]), 
                                      kind='linear', fill_value='extrapolate')
                    rr_interp = f_interp(time_interp)
                    
                    # Compute PSD
                    freqs, psd = welch(rr_interp, fs=interp_freq, nperseg=min(64, len(rr_interp)//4))
                    
                    fig.add_trace(
                        go.Scatter(x=freqs, y=psd, name='HRV PSD',
                                  line=dict(color='purple', width=2)),
                        row=2, col=1
                    )
                    
                    # Mark frequency bands
                    vlf_mask = (freqs >= 0.003) & (freqs <= 0.04)
                    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                    
                    if np.any(vlf_mask):
                        fig.add_trace(go.Scatter(x=freqs[vlf_mask], y=psd[vlf_mask], 
                                               fill='tozeroy', name='VLF', 
                                               fillcolor='rgba(255,0,0,0.3)'), row=2, col=1)
                    if np.any(lf_mask):
                        fig.add_trace(go.Scatter(x=freqs[lf_mask], y=psd[lf_mask], 
                                               fill='tozeroy', name='LF', 
                                               fillcolor='rgba(0,255,0,0.3)'), row=2, col=1)
                    if np.any(hf_mask):
                        fig.add_trace(go.Scatter(x=freqs[hf_mask], y=psd[hf_mask], 
                                               fill='tozeroy', name='HF', 
                                               fillcolor='rgba(0,0,255,0.3)'), row=2, col=1)
            except Exception as e:
                st.info(f"Frequency analysis: {str(e)}")
        
        # 5. Time domain trends
        if len(rr_ms) > 10:
            window = min(10, len(rr_ms)//3)
            moving_mean = pd.Series(rr_ms).rolling(window=window, center=True).mean()
            moving_std = pd.Series(rr_ms).rolling(window=window, center=True).std()
            
            fig.add_trace(
                go.Scatter(x=rr_indices, y=moving_mean, name='Moving Mean',
                          line=dict(color='blue', width=2)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=rr_indices, y=moving_std, name='Moving Std',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display HRV Features in organized tabs
        hrv_tab1, hrv_tab2, hrv_tab3 = st.tabs(["⏱️ Time Domain", "🌊 Frequency Domain", "🔄 Nonlinear"])
        
        with hrv_tab1:
            if data.time_features:
                display_features_table("Time Domain Features", data.time_features)
            else:
                st.info("No time domain features available")
        
        with hrv_tab2:
            if data.freq_features:
                display_features_table("Frequency Domain Features", data.freq_features)
            else:
                st.info("No frequency domain features available")
        
        with hrv_tab3:
            if data.nonlinear_features:
                display_features_table("Nonlinear Features", data.nonlinear_features)
            else:
                st.info("No nonlinear features available")

class DWTModule:
    """DWT filter analysis and visualization"""
    
    @staticmethod
    def create_plots(data: SignalData, plot_manager: PlotManager):
        """Create DWT filter analysis plots"""
        st.subheader("🔬 DWT Filter Bank Analysis")
        st.write("**Analysis of Discrete Wavelet Transform filter responses and characteristics**")
        
        # Generate filter responses
        try:
            if not hasattr(data.coef, 'frequency_responses') or not data.coef.frequency_responses:
                data.coef.store_filter_responses()
            
            freq_responses = data.coef.frequency_responses
            
            if freq_responses:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'DWT Filter Bank Frequency Response',
                        'Selected Filters (Respiratory & Vasometric)',
                        'Filter Magnitude Comparison',
                        'Frequency Band Analysis'
                    ],
                    vertical_spacing=0.15
                )
                
                freq_axis = freq_responses['freq_axis']
                Q_responses = freq_responses['Q_responses']
                
                # 1. All filter responses
                colors = px.colors.qualitative.Set3
                for i in range(min(8, len(Q_responses))):
                    if len(Q_responses[i]) > 0:
                        fig.add_trace(
                            go.Scatter(x=freq_axis, y=Q_responses[i], 
                                      name=f'Level {i+1}', 
                                      line=dict(color=colors[i % len(colors)], width=2)),
                            row=1, col=1
                        )
                
                # 2. Highlight selected filters
                if data.j_resp <= len(Q_responses) and len(Q_responses[data.j_resp-1]) > 0:
                    fig.add_trace(
                        go.Scatter(x=freq_axis, y=Q_responses[data.j_resp-1], 
                                  name=f'Respiratory (Level {data.j_resp})', 
                                  line=dict(color='green', width=4)),
                        row=1, col=2
                    )
                
                if data.j_vaso <= len(Q_responses) and len(Q_responses[data.j_vaso-1]) > 0:
                    fig.add_trace(
                        go.Scatter(x=freq_axis, y=Q_responses[data.j_vaso-1], 
                                  name=f'Vasometric (Level {data.j_vaso})', 
                                  line=dict(color='purple', width=4)),
                        row=1, col=2
                    )
                
                # 3. Magnitude comparison
                max_magnitudes = []
                center_freqs = []
                for i, response in enumerate(Q_responses):
                    if len(response) > 0:
                        max_mag = np.max(response)
                        max_idx = np.argmax(response)
                        center_freq = freq_axis[max_idx]
                        max_magnitudes.append(max_mag)
                        center_freqs.append(center_freq)
                    else:
                        max_magnitudes.append(0)
                        center_freqs.append(0)
                
                fig.add_trace(
                    go.Bar(x=[f'Level {i+1}' for i in range(len(max_magnitudes))], 
                          y=max_magnitudes, name='Peak Magnitude',
                          marker_color='blue'),
                    row=2, col=1
                )
                
                # 4. Frequency bands
                fig.add_trace(
                    go.Bar(x=[f'Level {i+1}' for i in range(len(center_freqs))], 
                          y=center_freqs, name='Center Frequency',
                          marker_color='orange'),
                    row=2, col=2
                )
                
                fig.update_layout(height=700, showlegend=True)
                fig.update_xaxes(title_text="Frequency (Hz)")
                fig.update_yaxes(title_text="Normalized Magnitude")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Filter characteristics table
                st.subheader("📋 Filter Characteristics")
                filter_data = []
                for i in range(len(max_magnitudes)):
                    if max_magnitudes[i] > 0:
                        filter_data.append({
                            'Level': i+1,
                            'Center Frequency (Hz)': f"{center_freqs[i]:.4f}",
                            'Peak Magnitude': f"{max_magnitudes[i]:.4f}",
                            'Usage': 'Respiratory' if i+1 == data.j_resp else 'Vasometric' if i+1 == data.j_vaso else 'Available'
                        })
                
                if filter_data:
                    df_filters = pd.DataFrame(filter_data)
                    st.dataframe(df_filters, use_container_width=True, hide_index=True)
                
            else:
                st.error("Could not generate filter responses")
                
        except Exception as e:
            st.error(f"Error generating DWT filter plots: {str(e)}")
            
        # DWT Theory explanation
        with st.expander("📖 Understanding DWT Levels"):
            st.write("""
            **Discrete Wavelet Transform (DWT) Level Selection:**
            
            - **Higher Levels (7-8)**: Capture lower frequency components
            - **Lower Levels (1-3)**: Capture higher frequency components
            - **Respiratory Analysis**: Typically uses Level 7 to capture breathing patterns (0.1-0.5 Hz)
            - **Vasometric Analysis**: Typically uses Level 8 to capture vascular oscillations (0.01-0.1 Hz)
            
            **Frequency Bands by Level:**
            - Level 1: Highest frequencies (noise, artifacts)
            - Level 4-5: Heart rate components (~1-2 Hz)
            - Level 7: Respiratory components (~0.2-0.5 Hz)
            - Level 8: Vasometric/thermoregulatory components (~0.01-0.1 Hz)
            """)

# ========== MAIN DISPLAY FUNCTION ==========
def display_results(signal_Hr, peaks_Hr, BPM, rr_intervals, resp_signal, resp_peaks, BrPM, 
                   peak_freq, resp_freq, resp_magnitude, vaso_signal, peak_vaso, vaso_freq, vaso_magnitude,
                   time_features, freq_features, nonlinear_features, fs, original_signal, coef, j_resp, j_vaso):
    """Display comprehensive analysis results with modular, synchronized plots"""
    
    st.markdown("---")
    st.header("📊 Comprehensive Signal Analysis Results")
    
    # Create data container
    results_dict = {
        'signal_Hr': signal_Hr, 'peaks_Hr': peaks_Hr, 'BPM': BPM, 'rr_intervals': rr_intervals,
        'resp_signal': resp_signal, 'resp_peaks': resp_peaks, 'BrPM': BrPM, 'peak_freq': peak_freq,
        'resp_freq': resp_freq, 'resp_magnitude': resp_magnitude,
        'vaso_signal': vaso_signal, 'peak_vaso': peak_vaso, 'vaso_freq': vaso_freq, 'vaso_magnitude': vaso_magnitude,
        'time_features': time_features, 'freq_features': freq_features, 'nonlinear_features': nonlinear_features,
        'original_signal': original_signal, 'coef': coef
    }
    
    data = SignalData(results_dict, fs, j_resp, j_vaso)
    plot_manager = PlotManager(data)
    
    # Display basic metrics
    MetricsDisplay.show_basic_metrics(data)
    
    # Create visualization options
    viz_option = st.radio(
        "📈 Visualization Mode:",
        ["🔗 Synchronized Signals", "📊 Detailed Tabs", "🧠 HRV Focus"],
        horizontal=True
    )
    
    if viz_option == "🔗 Synchronized Signals":
        create_synchronized_overview(data, plot_manager)
    elif viz_option == "📊 Detailed Tabs":
        create_detailed_tabs(data, plot_manager)
    else:
        create_hrv_focus(data, plot_manager)
    


def display_features_table(title, features):
    """Display HRV features with detailed descriptions and interpretations"""
    if not features:
        st.info(f"No {title.lower()} available")
        return
    
    # Convert to DataFrame with descriptions and interpretations
    feature_data = []
    for key, value in features.items():
        description = get_hrv_description(key)
        interpretation = get_hrv_interpretation(key, value)
        
        feature_data.append({
            "Parameter": key,
            "Value": f"{value:.4f}" if isinstance(value, float) else str(value),
            "Unit": get_unit(key),
            "Description": description,
            "Interpretation": interpretation,
        })
    
    df_features = pd.DataFrame(feature_data)
    
    # Display with expandable descriptions
    st.dataframe(df_features, use_container_width=True, hide_index=True)
    
    # Add expandable detailed explanations
    with st.expander(f"📖 Understanding {title}"):
        for item in feature_data:
            st.markdown(f"""
            **{item['Parameter']}** ({item['Unit']})
            - **Current Value:** {item['Value']} - *{item['Interpretation']}*
            - **Description:** {item['Description']}
            ---
            """)

def get_hrv_description(parameter):
    """Get detailed description for HRV parameters"""
    descriptions = {
        # Time Domain
        'SDNN_ms': 'Standard Deviation of NN intervals - Overall measure of heart rate variability',
        'SDANN_ms': 'Standard Deviation of Average NN intervals - Long-term HRV component',
        'RMSSD_ms': 'Root Mean Square of Successive Differences - Short-term HRV, reflects parasympathetic activity',
        'SDSD_ms': 'Standard Deviation of Successive Differences - Measure of beat-to-beat variability',
        'TINN_ms': 'Triangular Interpolation of NN intervals - Geometric measure of HRV distribution width',
        'NN50_count': 'Number of pairs of successive NNs that differ by more than 50ms',
        'pNN50_percent': 'Percentage of NN50 count divided by total number of NNs - Reflects parasympathetic tone',
        'CVNN_percent': 'Coefficient of Variation of NN intervals - Normalized measure of HRV',
        'CVSD_percent': 'Coefficient of Variation of Successive Differences - Normalized short-term HRV',
        
        # Frequency Domain
        'VLF': 'Very Low Frequency Power (0.003-0.04 Hz) - Reflects thermoregulation and long-term regulation',
        'LF': 'Low Frequency Power (0.04-0.15 Hz) - Mixed sympathetic and parasympathetic activity',
        'HF': 'High Frequency Power (0.15-0.4 Hz) - Primarily parasympathetic (vagal) activity',
        'LFnu': 'Low Frequency in normalized units - Relative LF power, often associated with sympathetic activity',
        'HFnu': 'High Frequency in normalized units - Relative HF power, reflects parasympathetic activity',
        'LF_HF_ratio': 'LF/HF Ratio - Balance between sympathetic and parasympathetic activity',
        
        # Nonlinear
        'SD1_ms': 'Poincaré plot standard deviation perpendicular to line of identity - Short-term variability',
        'SD2_ms': 'Poincaré plot standard deviation along line of identity - Long-term variability',
        'SD1_SD2_ratio': 'Ratio of SD1 to SD2 - Balance between short and long-term variability',
    }
    return descriptions.get(parameter, 'Heart rate variability parameter')

def get_hrv_interpretation(parameter, value):
    """Get interpretation of HRV parameter values"""
    if not isinstance(value, (int, float)):
        return "Non-numeric value"
    
    interpretations = {
        'SDNN_ms': get_sdnn_interpretation(value),
        'RMSSD_ms': get_rmssd_interpretation(value),
        'pNN50_percent': get_pnn50_interpretation(value),
        'LF': get_frequency_power_interpretation(value, 'LF'),
        'HF': get_frequency_power_interpretation(value, 'HF'),
        'LFnu': get_normalized_interpretation(value, 'LF'),
        'HFnu': get_normalized_interpretation(value, 'HF'),
        'SD1_ms': get_sd1_interpretation(value),
        'SD2_ms': get_sd2_interpretation(value),
    }
    
    return interpretations.get(parameter, get_general_interpretation(value))

def get_sdnn_interpretation(value):
    """Interpret SDNN values"""
    if value < 20:
        return "Very Low (< 20 ms)"
    elif value < 50:
        return "Low (20-50 ms)"
    elif value < 100:
        return "Normal (50-100 ms)"
    elif value < 150:
        return "High (100-150 ms)"
    else:
        return "Very High (> 150 ms)"

def get_rmssd_interpretation(value):
    """Interpret RMSSD values"""
    if value < 15:
        return "Very Low (< 15 ms)"
    elif value < 30:
        return "Low (15-30 ms)"
    elif value < 50:
        return "Normal (30-50 ms)"
    elif value < 80:
        return "High (50-80 ms)"
    else:
        return "Very High (> 80 ms)"

def get_pnn50_interpretation(value):
    """Interpret pNN50 values"""
    if value < 3:
        return "Very Low (< 3%)"
    elif value < 10:
        return "Low (3-10%)"
    elif value < 20:
        return "Normal (10-20%)"
    elif value < 35:
        return "High (20-35%)"
    else:
        return "Very High (> 35%)"

def get_frequency_power_interpretation(value, band):
    """Interpret frequency domain power values"""
    if band == 'LF':
        if value < 100:
            return "Low (< 100 ms²)"
        elif value < 500:
            return "Normal (100-500 ms²)"
        else:
            return "High (> 500 ms²)"
    elif band == 'HF':
        if value < 50:
            return "Low (< 50 ms²)"
        elif value < 200:
            return "Normal (50-200 ms²)"
        else:
            return "High (> 200 ms²)"
    return "Within normal range"

def get_normalized_interpretation(value, band):
    """Interpret normalized frequency values"""
    if band == 'LF':
        if value < 30:
            return "Low (< 30%)"
        elif value < 70:
            return "Normal (30-70%)"
        else:
            return "High (> 70%)"
    elif band == 'HF':
        if value < 20:
            return "Low (< 20%)"
        elif value < 50:
            return "Normal (20-50%)"
        else:
            return "High (> 50%)"
    return "Within normal range"

def get_sd1_interpretation(value):
    """Interpret SD1 values"""
    if value < 10:
        return "Very Low (< 10 ms)"
    elif value < 25:
        return "Low (10-25 ms)"
    elif value < 45:
        return "Normal (25-45 ms)"
    else:
        return "High (> 45 ms)"

def get_sd2_interpretation(value):
    """Interpret SD2 values"""
    if value < 20:
        return "Very Low (< 20 ms)"
    elif value < 60:
        return "Low (20-60 ms)"
    elif value < 120:
        return "Normal (60-120 ms)"
    else:
        return "High (> 120 ms)"

def get_general_interpretation(value):
    """General interpretation for unlisted parameters"""
    return "Within measured range"

def get_clinical_meaning(parameter, value):
    """Get standard range description"""
    if not isinstance(value, (int, float)):
        return "Non-numeric value"
    
    range_descriptions = {
        'SDNN_ms': "Standard ranges: < 20ms (Very Low), 20-50ms (Low), 50-100ms (Normal), 100-150ms (High), > 150ms (Very High)",
        'RMSSD_ms': "Standard ranges: < 15ms (Very Low), 15-30ms (Low), 30-50ms (Normal), 50-80ms (High), > 80ms (Very High)",
        'pNN50_percent': "Standard ranges: < 3% (Very Low), 3-10% (Low), 10-20% (Normal), 20-35% (High), > 35% (Very High)",
        'LF': "Standard ranges: < 100ms² (Low), 100-500ms² (Normal), > 500ms² (High)",
        'HF': "Standard ranges: < 50ms² (Low), 50-200ms² (Normal), > 200ms² (High)",
        'LFnu': "Standard ranges: < 30% (Low), 30-70% (Normal), > 70% (High)",
        'HFnu': "Standard ranges: < 20% (Low), 20-50% (Normal), > 50% (High)",
        'SD1_ms': "Standard ranges: < 10ms (Very Low), 10-25ms (Low), 25-45ms (Normal), > 45ms (High)",
        'SD2_ms': "Standard ranges: < 20ms (Very Low), 20-60ms (Low), 60-120ms (Normal), > 120ms (High)",
    }
    
    return range_descriptions.get(parameter, "No standard ranges defined for this parameter")



# ========== LEGACY FUNCTIONS (keeping for compatibility) ==========
def create_signal_processing_plots_legacy(original_signal, processed_signal, fs):
    """Create signal preprocessing visualization plots"""
    st.subheader("🔧 Signal Preprocessing")
    st.write("**Shows the effect of preprocessing steps on the original signal**")
    
    # Create comparison plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Original Signal', 'Processed Signal (Centered + Bandpass Filtered)'],
        vertical_spacing=0.15
    )
    
    time_orig = np.arange(len(original_signal)) / fs
    time_proc = np.arange(len(processed_signal)) / fs
    
    # Original signal
    fig.add_trace(
        go.Scatter(x=time_orig, y=original_signal, name='Original', 
                  line=dict(color='gray', width=1)),
        row=1, col=1
    )
    
    # Processed signal
    fig.add_trace(
        go.Scatter(x=time_proc, y=processed_signal, name='Processed', 
                  line=dict(color='blue', width=1)),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Amplitude")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Mean", f"{np.mean(original_signal):.4f}")
        st.metric("Original Std", f"{np.std(original_signal):.4f}")
    with col2:
        st.metric("Processed Mean", f"{np.mean(processed_signal):.4f}")
        st.metric("Processed Std", f"{np.std(processed_signal):.4f}")
    with col3:
        st.metric("Signal Length", f"{len(original_signal)} samples")
        st.metric("Duration", f"{len(original_signal)/fs:.2f} seconds")

def create_heart_rate_plots(signal_Hr, peaks_Hr, BPM, rr_intervals, fs):
    """Create detailed heart rate analysis plots"""
    st.subheader("💓 Heart Rate Analysis")
    st.write("**Detailed analysis of heart rate detection and variability**")
    
    # Create subplots for HR analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Heart Rate Signal with Peak Detection',
            'RR Interval Tachogram',
            'RR Interval Distribution',
            'Beat-to-Beat Interval Analysis'
        ],
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        vertical_spacing=0.15
    )
    
    time_hr = np.arange(len(signal_Hr)) / fs
    
    # 1. HR Signal with peaks
    fig.add_trace(
        go.Scatter(x=time_hr, y=signal_Hr, name='HR Signal', 
                  line=dict(color='red', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_hr[peaks_Hr], y=signal_Hr[peaks_Hr], 
                  mode='markers', name='Detected Peaks', 
                  marker=dict(color='blue', size=8, symbol='triangle-up')),
        row=1, col=1
    )
    
    # 2. RR Interval tachogram
    if len(rr_intervals) > 0:
        rr_times = peaks_Hr[1:] / fs  # Time of each RR interval
        rr_ms = rr_intervals * 1000  # Convert to milliseconds
        
        fig.add_trace(
            go.Scatter(x=rr_times, y=rr_ms, mode='lines+markers',
                      name='RR Intervals', line=dict(color='green', width=2),
                      marker=dict(size=4)),
            row=2, col=1
        )
        
        # 3. RR Interval histogram
        fig.add_trace(
            go.Histogram(x=rr_ms, nbinsx=20, name='RR Distribution',
                        marker_color='orange', opacity=0.7),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="RR Interval (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="RR Interval (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # HR Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average BPM", f"{BPM:.1f}")
    with col2:
        st.metric("Peak Count", f"{len(peaks_Hr)}")
    with col3:
        if len(rr_intervals) > 0:
            st.metric("Avg RR (ms)", f"{np.mean(rr_intervals)*1000:.1f}")
    with col4:
        if len(rr_intervals) > 0:
            st.metric("RR Std (ms)", f"{np.std(rr_intervals)*1000:.1f}")

def create_respiratory_plots(resp_signal, resp_peaks, BrPM, peak_freq, resp_freq, resp_magnitude, fs):
    """Create detailed respiratory analysis plots"""
    st.subheader("🫁 Respiratory Signal Analysis")
    st.write("**Analysis of breathing patterns and frequency characteristics**")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Respiratory Signal with Peak Detection',
            'Frequency Spectrum',
            'Respiratory Rate Over Time',
            'Peak Detection Statistics'
        ],
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        vertical_spacing=0.15
    )
    
    time_resp = np.arange(len(resp_signal)) / fs
    
    # 1. Respiratory signal with peaks
    fig.add_trace(
        go.Scatter(x=time_resp, y=resp_signal, name='Respiratory Signal', 
                  line=dict(color='green', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resp[resp_peaks], y=resp_signal[resp_peaks], 
                  mode='markers', name='Detected Peaks', 
                  marker=dict(color='orange', size=8, symbol='circle')),
        row=1, col=1
    )
    
    # 2. Frequency spectrum
    fig.add_trace(
        go.Scatter(x=resp_freq, y=resp_magnitude, name='Frequency Spectrum',
                  line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    # Mark peak frequency
    peak_idx = np.argmax(resp_magnitude[1:]) + 1  # Ignore DC
    fig.add_trace(
        go.Scatter(x=[resp_freq[peak_idx]], y=[resp_magnitude[peak_idx]],
                  mode='markers', name='Peak Frequency',
                  marker=dict(color='red', size=12, symbol='star')),
        row=2, col=1
    )
    
    # 3. Respiratory rate calculation over time windows
    if len(resp_peaks) > 2:
        window_size = min(len(resp_peaks)//3, 10)  # Adaptive window
        breathing_rates = []
        window_times = []
        
        for i in range(0, len(resp_peaks)-window_size, window_size//2):
            window_peaks = resp_peaks[i:i+window_size]
            window_duration = (window_peaks[-1] - window_peaks[0]) / fs
            if window_duration > 0:
                rate = (len(window_peaks)-1) / window_duration * 60
                breathing_rates.append(rate)
                window_times.append(window_peaks[len(window_peaks)//2] / fs)
        
        if breathing_rates:
            fig.add_trace(
                go.Scatter(x=window_times, y=breathing_rates, mode='lines+markers',
                          name='Breathing Rate', line=dict(color='purple', width=2)),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Breathing Rate (BrPM)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Respiratory Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Breathing Rate", f"{BrPM:.1f} BrPM")
    with col2:
        st.metric("Peak Frequency", f"{peak_freq:.4f} Hz")
    with col3:
        st.metric("Breath Count", f"{len(resp_peaks)}")
    with col4:
        duration = len(resp_signal) / fs
        st.metric("Analysis Duration", f"{duration:.1f} sec")

def create_vasometric_plots(vaso_signal, peak_vaso, vaso_freq, vaso_magnitude, fs):
    """Create vasometric signal analysis plots"""
    st.subheader("🩸 Vasometric Signal Analysis")
    st.write("**Analysis of vascular oscillations and blood flow patterns**")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Vasometric Signal',
            'Frequency Spectrum',
            'Power Spectral Density',
            'Signal Statistics'
        ],
        vertical_spacing=0.15
    )
    
    time_vaso = np.arange(len(vaso_signal)) / fs
    
    # 1. Vasometric signal
    fig.add_trace(
        go.Scatter(x=time_vaso, y=vaso_signal, name='Vasometric Signal',
                  line=dict(color='purple', width=1)),
        row=1, col=1
    )
    
    # 2. Frequency spectrum
    fig.add_trace(
        go.Scatter(x=vaso_freq, y=vaso_magnitude, name='Magnitude Spectrum',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    
    # Mark peak frequency
    peak_idx = np.argmax(vaso_magnitude[1:]) + 1
    fig.add_trace(
        go.Scatter(x=[vaso_freq[peak_idx]], y=[vaso_magnitude[peak_idx]],
                  mode='markers', name='Peak Frequency',
                  marker=dict(color='red', size=12, symbol='star')),
        row=1, col=2
    )
    
    # 3. Power spectral density using Welch's method
    try:
        freqs, psd = welch(vaso_signal, fs=fs, nperseg=min(256, len(vaso_signal)//4))
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, name='PSD (Welch)',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
    except:
        st.info("PSD calculation requires longer signal")
    
    # 4. Signal envelope and characteristics
    from scipy.signal import hilbert
    try:
        analytic_signal = hilbert(vaso_signal)
        envelope = np.abs(analytic_signal)
        fig.add_trace(
            go.Scatter(x=time_vaso, y=envelope, name='Signal Envelope',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_vaso, y=vaso_signal, name='Original',
                      line=dict(color='purple', width=1, dash='dot')),
            row=2, col=2
        )
    except:
        pass
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Vasometric Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Peak Frequency", f"{peak_vaso:.4f} Hz")
    with col2:
        st.metric("Signal RMS", f"{np.sqrt(np.mean(vaso_signal**2)):.4f}")
    with col3:
        st.metric("Signal Range", f"{np.ptp(vaso_signal):.4f}")
    with col4:
        st.metric("Zero Crossings", f"{len(np.where(np.diff(np.sign(vaso_signal)))[0])}")

def create_hrv_analysis_plots(rr_intervals, time_features, freq_features, nonlinear_features, fs):
    """Create comprehensive HRV analysis plots"""
    st.subheader("🧠 Heart Rate Variability (HRV) Analysis")
    st.write("**Comprehensive analysis of heart rate variability patterns**")
    
    if len(rr_intervals) < 2:
        st.warning("⚠️ Insufficient RR intervals for HRV analysis")
        return
    
    # Create HRV plots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'RR Interval Tachogram',
            'RR Interval Histogram', 
            'Poincaré Plot',
            'Frequency Domain Analysis',
            'Time Domain Trends',
            'Statistical Summary'
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    rr_ms = rr_intervals * 1000  # Convert to milliseconds
    rr_indices = np.arange(len(rr_ms))
    
    # 1. RR Tachogram
    fig.add_trace(
        go.Scatter(x=rr_indices, y=rr_ms, mode='lines+markers',
                  name='RR Intervals', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # 2. RR Histogram
    fig.add_trace(
        go.Histogram(x=rr_ms, nbinsx=20, name='Distribution',
                    marker_color='green', opacity=0.7),
        row=1, col=2
    )
    
    # 3. Poincaré Plot
    if len(rr_ms) > 1:
        rr1 = rr_ms[:-1]
        rr2 = rr_ms[1:]
        fig.add_trace(
            go.Scatter(x=rr1, y=rr2, mode='markers',
                      name='Poincaré', marker=dict(color='red', size=4)),
            row=1, col=3
        )
    
    # 4. Frequency Domain (if available)
    if freq_features:
        try:
            # Interpolate RR intervals for frequency analysis
            time_rr = np.cumsum(np.concatenate([[0], rr_intervals]))
            interp_freq = 4.0  # 4 Hz interpolation
            time_interp = np.arange(0, time_rr[-1], 1/interp_freq)
            
            if len(time_rr) > 1 and len(time_interp) > 10:
                f_interp = interp1d(time_rr, np.concatenate([[rr_ms[0]], rr_ms]), 
                                  kind='linear', fill_value='extrapolate')
                rr_interp = f_interp(time_interp)
                
                # Compute PSD
                freqs, psd = welch(rr_interp, fs=interp_freq, nperseg=min(64, len(rr_interp)//4))
                
                fig.add_trace(
                    go.Scatter(x=freqs, y=psd, name='HRV PSD',
                              line=dict(color='purple', width=2)),
                    row=2, col=1
                )
                
                # Mark frequency bands
                vlf_mask = (freqs >= 0.003) & (freqs <= 0.04)
                lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                
                if np.any(vlf_mask):
                    fig.add_trace(go.Scatter(x=freqs[vlf_mask], y=psd[vlf_mask], 
                                           fill='tozeroy', name='VLF', 
                                           fillcolor='rgba(255,0,0,0.3)'), row=2, col=1)
                if np.any(lf_mask):
                    fig.add_trace(go.Scatter(x=freqs[lf_mask], y=psd[lf_mask], 
                                           fill='tozeroy', name='LF', 
                                           fillcolor='rgba(0,255,0,0.3)'), row=2, col=1)
                if np.any(hf_mask):
                    fig.add_trace(go.Scatter(x=freqs[hf_mask], y=psd[hf_mask], 
                                           fill='tozeroy', name='HF', 
                                           fillcolor='rgba(0,0,255,0.3)'), row=2, col=1)
        except Exception as e:
            st.info(f"Frequency analysis: {str(e)}")
    
    # 5. Time domain trends
    if len(rr_ms) > 10:
        # Moving statistics
        window = min(10, len(rr_ms)//3)
        moving_mean = pd.Series(rr_ms).rolling(window=window, center=True).mean()
        moving_std = pd.Series(rr_ms).rolling(window=window, center=True).std()
        
        fig.add_trace(
            go.Scatter(x=rr_indices, y=moving_mean, name='Moving Mean',
                      line=dict(color='blue', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=rr_indices, y=moving_std, name='Moving Std',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display HRV Features in organized tabs
    hrv_tab1, hrv_tab2, hrv_tab3 = st.tabs(["⏱️ Time Domain", "🌊 Frequency Domain", "🔄 Nonlinear"])
    
    with hrv_tab1:
        if time_features:
            display_features_table("Time Domain Features", time_features)
        else:
            st.info("No time domain features available")
    
    with hrv_tab2:
        if freq_features:
            display_features_table("Frequency Domain Features", freq_features)
        else:
            st.info("No frequency domain features available")
    
    with hrv_tab3:
        if nonlinear_features:
            display_features_table("Nonlinear Features", nonlinear_features)
        else:
            st.info("No nonlinear features available")

def create_dwt_filter_plots(coef, fs, j_resp, j_vaso):
    """Create DWT filter analysis plots"""
    st.subheader("🔬 DWT Filter Bank Analysis")
    st.write("**Analysis of Discrete Wavelet Transform filter responses and characteristics**")
    
    # Generate filter responses
    try:
        if not hasattr(coef, 'frequency_responses') or not coef.frequency_responses:
            coef.store_filter_responses()
        
        freq_responses = coef.frequency_responses
        
        if freq_responses:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'DWT Filter Bank Frequency Response',
                    'Selected Filters (Respiratory & Vasometric)',
                    'Filter Magnitude Comparison',
                    'Frequency Band Analysis'
                ],
                vertical_spacing=0.15
            )
            
            freq_axis = freq_responses['freq_axis']
            Q_responses = freq_responses['Q_responses']
            
            # 1. All filter responses
            colors = px.colors.qualitative.Set3
            for i in range(min(8, len(Q_responses))):
                if len(Q_responses[i]) > 0:
                    fig.add_trace(
                        go.Scatter(x=freq_axis, y=Q_responses[i], 
                                  name=f'Level {i+1}', 
                                  line=dict(color=colors[i % len(colors)], width=2)),
                        row=1, col=1
                    )
            
            # 2. Highlight selected filters
            if j_resp <= len(Q_responses) and len(Q_responses[j_resp-1]) > 0:
                fig.add_trace(
                    go.Scatter(x=freq_axis, y=Q_responses[j_resp-1], 
                              name=f'Respiratory (Level {j_resp})', 
                              line=dict(color='green', width=4)),
                    row=1, col=2
                )
            
            if j_vaso <= len(Q_responses) and len(Q_responses[j_vaso-1]) > 0:
                fig.add_trace(
                    go.Scatter(x=freq_axis, y=Q_responses[j_vaso-1], 
                              name=f'Vasometric (Level {j_vaso})', 
                              line=dict(color='purple', width=4)),
                    row=1, col=2
                )
            
            # 3. Magnitude comparison
            max_magnitudes = []
            center_freqs = []
            for i, response in enumerate(Q_responses):
                if len(response) > 0:
                    max_mag = np.max(response)
                    max_idx = np.argmax(response)
                    center_freq = freq_axis[max_idx]
                    max_magnitudes.append(max_mag)
                    center_freqs.append(center_freq)
                else:
                    max_magnitudes.append(0)
                    center_freqs.append(0)
            
            fig.add_trace(
                go.Bar(x=[f'Level {i+1}' for i in range(len(max_magnitudes))], 
                      y=max_magnitudes, name='Peak Magnitude',
                      marker_color='blue'),
                row=2, col=1
            )
            
            # 4. Frequency bands
            fig.add_trace(
                go.Bar(x=[f'Level {i+1}' for i in range(len(center_freqs))], 
                      y=center_freqs, name='Center Frequency',
                      marker_color='orange'),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=True)
            fig.update_xaxes(title_text="Frequency (Hz)")
            fig.update_yaxes(title_text="Normalized Magnitude")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Filter characteristics table
            st.subheader("📋 Filter Characteristics")
            filter_data = []
            for i in range(len(max_magnitudes)):
                if max_magnitudes[i] > 0:
                    filter_data.append({
                        'Level': i+1,
                        'Center Frequency (Hz)': f"{center_freqs[i]:.4f}",
                        'Peak Magnitude': f"{max_magnitudes[i]:.4f}",
                        'Usage': 'Respiratory' if i+1 == j_resp else 'Vasometric' if i+1 == j_vaso else 'Available'
                    })
            
            if filter_data:
                df_filters = pd.DataFrame(filter_data)
                st.dataframe(df_filters, use_container_width=True, hide_index=True)
            
        else:
            st.error("Could not generate filter responses")
            
    except Exception as e:
        st.error(f"Error generating DWT filter plots: {str(e)}")
        
    # DWT Theory explanation
    with st.expander("📖 Understanding DWT Levels"):
        st.write("""
        **Discrete Wavelet Transform (DWT) Level Selection:**
        
        - **Higher Levels (7-8)**: Capture lower frequency components
        - **Lower Levels (1-3)**: Capture higher frequency components
        - **Respiratory Analysis**: Typically uses Level 7 to capture breathing patterns (0.1-0.5 Hz)
        - **Vasometric Analysis**: Typically uses Level 8 to capture vascular oscillations (0.01-0.1 Hz)
        
        **Frequency Bands by Level:**
        - Level 1: Highest frequencies (noise, artifacts)
        - Level 4-5: Heart rate components (~1-2 Hz)
        - Level 7: Respiratory components (~0.2-0.5 Hz)
        - Level 8: Vasometric/thermoregulatory components (~0.01-0.1 Hz)
        """)

# ========== VISUALIZATION MODES ==========
def create_synchronized_overview(data: SignalData, plot_manager: PlotManager):
    """Create synchronized overview of all signals with linked zooming"""
    st.subheader("🔗 Synchronized Signal Analysis")
    st.write("**All signals are synchronized - zoom on one affects all time-domain plots**")
    
    # Create synchronized figure with all signals
    fig = plot_manager.create_synchronized_figure(
        rows=4, cols=1,
        subplot_titles=[
            f'Original vs Processed Signal',
            f'Heart Rate Signal - BPM: {data.BPM:.2f}',
            f'Respiratory Signal - BrPM: {data.BrPM:.2f}',
            f'Vasometric Signal - Peak Freq: {data.peak_vaso:.4f} Hz'
        ]
    )
    
    # 1. Original vs Processed
    plot_manager.add_time_series_trace(
        fig, data.time_orig, data.original_signal, 'Original Signal',
        1, 1, line=dict(color='gray', width=1)
    )
    plot_manager.add_time_series_trace(
        fig, data.time_hr, data.signal_Hr, 'Processed Signal',
        1, 1, line=dict(color='blue', width=1.5)
    )
    
    # 2. Heart Rate with peaks
    plot_manager.add_time_series_trace(
        fig, data.time_hr, data.signal_Hr, 'HR Signal',
        2, 1, line=dict(color='red', width=1.5)
    )
    plot_manager.add_peaks_trace(
        fig, data.time_hr[data.peaks_Hr], data.signal_Hr[data.peaks_Hr], 'HR Peaks',
        2, 1, marker=dict(color='darkred', size=8, symbol='triangle-up')
    )
    
    # 3. Respiratory with peaks
    plot_manager.add_time_series_trace(
        fig, data.time_resp, data.resp_signal, 'Respiratory Signal',
        3, 1, line=dict(color='green', width=1.5)
    )
    plot_manager.add_peaks_trace(
        fig, data.time_resp[data.resp_peaks], data.resp_signal[data.resp_peaks], 'Resp Peaks',
        3, 1, marker=dict(color='orange', size=6, symbol='circle')
    )
    
    # 4. Vasometric
    plot_manager.add_time_series_trace(
        fig, data.time_vaso, data.vaso_signal, 'Vasometric Signal',
        4, 1, line=dict(color='purple', width=1.5)
    )
    
    # Update layout with synchronized zoom
    fig.update_layout(
        height=800,
        showlegend=True,
        title="Synchronized Signal Analysis (Linked Zoom)",
        xaxis4_title="Time (seconds)"  # Only label bottom x-axis
    )
    
    # Update all y-axis labels
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="HR Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Resp Amplitude", row=3, col=1)
    fig.update_yaxes(title_text="Vaso Amplitude", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed metrics
    with st.expander("📊 Detailed Signal Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            MetricsDisplay.show_signal_stats(data.signal_Hr, data.fs, "Heart Rate")
        with col2:
            MetricsDisplay.show_signal_stats(data.resp_signal, data.fs, "Respiratory")

def create_detailed_tabs(data: SignalData, plot_manager: PlotManager):
    """Create detailed tabbed analysis"""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Signal Processing", 
        "💓 Heart Rate Analysis", 
        "🫁 Respiratory Analysis",
        "🩸 Vasometric Analysis", 
        "🧠 HRV Analysis",
        "🔬 DWT Filter Analysis"
    ])
    
    with tab1:
        SignalProcessingModule.create_plots(data, plot_manager)
    
    with tab2:
        HeartRateModule.create_plots(data, plot_manager)
    
    with tab3:
        RespiratoryModule.create_plots(data, plot_manager)
    
    with tab4:
        VasometricModule.create_plots(data, plot_manager)
    
    with tab5:
        HRVModule.create_plots(data, plot_manager)
    
    with tab6:
        DWTModule.create_plots(data, plot_manager)

def create_hrv_focus(data: SignalData, plot_manager: PlotManager):
    """Create HRV-focused analysis"""
    st.subheader("🧠 HRV-Focused Analysis")
    
    # Quick overview
    col1, col2 = st.columns(2)
    with col1:
        # RR intervals overview
        fig_rr = make_subplots(rows=1, cols=1, subplot_titles=['RR Interval Tachogram'])
        rr_ms = data.rr_intervals * 1000
        rr_times = data.peaks_Hr[1:] / data.fs
        
        fig_rr.add_trace(
            go.Scatter(x=rr_times, y=rr_ms, mode='lines+markers',
                      name='RR Intervals', line=dict(color='blue', width=2))
        )
        fig_rr.update_layout(height=300)
        st.plotly_chart(fig_rr, use_container_width=True)
    
    with col2:
        # Feature summary
        st.write("**Key HRV Metrics:**")
        if data.time_features:
            for key, value in list(data.time_features.items())[:5]:
                st.metric(key.replace('_', ' '), f"{value:.2f}", get_unit(key))
    
    # Detailed HRV analysis
    HRVModule.create_detailed_analysis(data, plot_manager)

def get_unit(parameter_name):
    """Get appropriate unit for parameter"""
    unit_map = {
        'SDNN_ms': 'ms',
        'SDANN_ms': 'ms',
        'RMSSD_ms': 'ms',
        'SDSD_ms': 'ms',
        'TINN_ms': 'ms',
        'SD1_ms': 'ms',
        'SD2_ms': 'ms',
        'NN50_count': 'count',
        'pNN50_percent': '%',
        'CVNN_percent': '%',
        'CVSD_percent': '%',
        'LFnu': '%',
        'HFnu': '%',
        'VLF': 'ms²',
        'LF': 'ms²',
        'HF': 'ms²'
    }
    return unit_map.get(parameter_name, '')

if __name__ == "__main__":
    main()