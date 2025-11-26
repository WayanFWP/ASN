import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import sys
import io
from contextlib import redirect_stdout, redirect_stderr  # Fixed import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.style as mplstyle

# Import your modules
from Coeficient import Coeficient
from Utils import *
from HRV import HRV as feature
from Analyze import Respiratory, Vasometric, HeartRate

class SignalAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analysis Pipeline")
        self.root.geometry("1400x900")
        
        # Configure matplotlib to use non-interactive backend
        plt.ioff()
        mplstyle.use('default')
        
        # Analysis components
        self.coef = None
        self.HR = None
        self.Resp = None
        self.Vaso = None
        
        # Results storage
        self.results = {}
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Create main frame with tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize GUI mode for coef
        self.gui_mode = True
        
        self.create_input_tab()
        self.create_results_tab()
        self.create_plots_tab()
        
        self.setup_output_monitoring()
        
    def create_input_tab(self):
        # Input and Configuration Tab
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Configuration & Analysis")
        
        # File selection section
        file_frame = ttk.LabelFrame(self.input_frame, text="Data File", padding="10")
        file_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=50)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        file_frame.columnconfigure(1, weight=1)
        
        # Column selection
        ttk.Label(file_frame, text="Signal Column:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(file_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        
        # Parameters section
        params_frame = ttk.LabelFrame(self.input_frame, text="Analysis Parameters", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Sampling frequency
        ttk.Label(params_frame, text="Original Sampling Frequency (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.fs_var = tk.DoubleVar(value=50.0)
        ttk.Entry(params_frame, textvariable=self.fs_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Downsampling factor
        ttk.Label(params_frame, text="Downsampling Factor:").grid(row=0, column=2, sticky=tk.W)
        self.factor_var = tk.DoubleVar(value=1.15)
        ttk.Entry(params_frame, textvariable=self.factor_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Filter parameters
        ttk.Label(params_frame, text="Bandpass Low (Hz):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.bp_low_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.bp_low_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 20), pady=(5, 0))
        
        ttk.Label(params_frame, text="Bandpass High (Hz):").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        self.bp_high_var = tk.DoubleVar(value=45.0)
        ttk.Entry(params_frame, textvariable=self.bp_high_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=(5, 0))
        
        # DWT levels
        ttk.Label(params_frame, text="Respiratory DWT Level:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.j_resp_var = tk.IntVar(value=7)
        ttk.Entry(params_frame, textvariable=self.j_resp_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=(5, 20), pady=(5, 0))
        
        ttk.Label(params_frame, text="Vasometric DWT Level:").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))
        self.j_vaso_var = tk.IntVar(value=8)
        ttk.Entry(params_frame, textvariable=self.j_vaso_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=5, pady=(5, 0))
        
        # Control buttons
        control_frame = ttk.Frame(self.input_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.run_button = ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear Output", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Export Plots", command=self.export_plots).pack(side=tk.LEFT, padx=5)
        
        # Add button to show DWT filter response
        ttk.Button(control_frame, text="Show DWT Filters", command=self.show_dwt_filters).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.input_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Output section (smaller now)
        output_frame = ttk.LabelFrame(self.input_frame, text="Analysis Output", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, width=80)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.input_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, padx=10, pady=(5, 10))
    
    def create_results_tab(self):
        # Results Tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Create treeview for results
        results_tree_frame = ttk.LabelFrame(self.results_frame, text="Analysis Results", padding="5")
        results_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview with scrollbars
        tree_container = ttk.Frame(results_tree_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_tree = ttk.Treeview(tree_container, columns=('Value', 'Unit'), show='tree headings')
        self.results_tree.heading('#0', text='Parameter')
        self.results_tree.heading('Value', text='Value')
        self.results_tree.heading('Unit', text='Unit')
        
        # Column configuration
        self.results_tree.column('#0', width=250)
        self.results_tree.column('Value', width=150)
        self.results_tree.column('Unit', width=100)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
    def create_plots_tab(self):
        # Plots Tab
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Signal Plots")
        
        # Create plot control frame
        plot_control_frame = ttk.Frame(self.plots_frame)
        plot_control_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Button(plot_control_frame, text="Refresh Plots", command=self.refresh_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_control_frame, text="Save Plots", command=self.save_plots).pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib figure with higher DPI for better quality
        self.fig = Figure(figsize=(14, 10), dpi=100, tight_layout=True)
        self.fig.patch.set_facecolor('white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add navigation toolbar
        self.toolbar_frame = ttk.Frame(self.plots_frame)
        self.toolbar_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
            self.load_file_columns(filename)
    
    def load_file_columns(self, filename):
        try:
            df = pd.read_csv(filename, nrows=1)  # Read only first row to get columns
            columns = df.columns.tolist()
            self.column_combo['values'] = columns
            if len(columns) > 1:
                self.column_var.set(columns[1])  # Default to second column
            self.log_output(f"Loaded file with columns: {columns}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def show_dwt_filters(self):
        """Show DWT filter responses in a separate window"""
        try:
            fs = self.fs_var.get()
            factor = self.factor_var.get()
            fs_effective = fs / factor
            
            # Create coefficient object for filter display
            coef_display = Coeficient(fs_effective)
            coef_display.gui_mode = True  # Set GUI mode to prevent popup plots
            
            # Create a new window for DWT filters
            dwt_window = tk.Toplevel(self.root)
            dwt_window.title("DWT Filter Responses")
            dwt_window.geometry("1000x700")
            
            # Create figure for DWT filters
            dwt_fig = Figure(figsize=(12, 8), dpi=100)
            dwt_fig.patch.set_facecolor('white')
            
            # Create canvas
            dwt_canvas = FigureCanvasTkAgg(dwt_fig, master=dwt_window)
            dwt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add navigation toolbar
            dwt_toolbar_frame = ttk.Frame(dwt_window)
            dwt_toolbar_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            dwt_toolbar = NavigationToolbar2Tk(dwt_canvas, dwt_toolbar_frame)
            dwt_toolbar.update()
            
            # Generate DWT filter responses without showing popup plots
            self.log_output("Generating DWT filter responses...")
            
            # Initialize the filters first
            coef_display.initialize_qj_filter()
            
            # Call the method to generate filter responses and plot them in our figure
            coef_display.plot_filter_responses(dwt_fig)
            dwt_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show DWT filters: {str(e)}")
            self.log_output(f"Error showing DWT filters: {str(e)}")
    
    def run_analysis(self):
        """Run the signal analysis pipeline in a separate thread"""
        if not self.file_var.get():
            messagebox.showerror("Error", "Please select a CSV file")
            return
        
        if not self.column_var.get():
            messagebox.showerror("Error", "Please select a signal column")
            return
            
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Running analysis...")
        self.progress_var.set(0)
        
        # Clear previous output and plots
        self.output_text.delete(1.0, tk.END)
        self.clear_results_tree()
        self.clear_plots()
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.execute_analysis, daemon=True)
        self.analysis_thread.start()
    
    def execute_analysis(self):
        """Execute the actual signal analysis pipeline"""
        try:
            # Load data
            self.log_output("Loading data...")
            self.update_progress(5)
            
            load_file = pd.read_csv(self.file_var.get())
            selected_signal = self.column_var.get()
            
            # Get parameters
            fs = self.fs_var.get()
            factor = self.factor_var.get()
            fs_effective = fs / factor
            
            self.log_output(f"Original FS: {fs} Hz, Factor: {factor}, Effective FS: {fs_effective:.2f} Hz")
            
            # Initialize components
            self.log_output("Initializing analysis components...")
            self.update_progress(10)
            
            # IMPORTANT: Initialize Coeficient with GUI mode to disable plotting
            self.coef = Coeficient(fs_effective)
            self.coef.gui_mode = True  # Set GUI mode to disable popup plots
            self.coef.initialize_qj_filter()
            
            self.HR = HeartRate(fs_effective)
            self.Resp = Respiratory(fs_effective)
            self.Vaso = Vasometric(fs_effective)
            
            # Process signal
            self.log_output("Processing signal...")
            self.update_progress(20)
            
            signal = downSample(load_file[selected_signal].values, factor)
            
            # Preprocessing
            self.log_output("Preprocessing signal...")
            self.update_progress(30)
            
            mean_signal = np.mean(signal)
            signal = signal - mean_signal  # Centering
            signal = BPF(signal, self.bp_low_var.get(), self.bp_high_var.get(), fs_effective)
            
            # Heart Rate Analysis
            self.log_output("Analyzing heart rate...")
            self.update_progress(40)
            
            signal_Hr, peaks_Hr, BPM = self.HR.analyze(signal)
            rr_intervals = np.diff(peaks_Hr / fs_effective)
            self.log_output(f"BPM: {BPM:.2f}")
            
            # Apply DWT and Respiratory Analysis
            self.log_output("Performing DWT and respiratory analysis...")
            self.update_progress(60)
            
            J_Resp = self.j_resp_var.get()
            signal_DWT = self.coef.applying(signal, specific_j=J_Resp)
            resp_data, resp_peaks, BrPM = self.Resp.analyze(signal_DWT[J_Resp])
            freq, magnitude, peak_freq, peak_mag = self.Resp.get_freq()
            self.log_output(f"Respiratory frequency: {peak_freq:.4f} Hz, BrPM: {BrPM:.2f}")
            
            # Vasometric Analysis
            self.log_output("Performing vasometric analysis...")
            self.update_progress(80)
            
            J_vaso = self.j_vaso_var.get()
            signal_dwtvaso = self.coef.applying(signal, specific_j=J_vaso)
            vaso_freq, vaso_mag, peak_vaso, vaso_peak_mag = self.Vaso.analyze(signal_dwtvaso[J_vaso])
            self.log_output(f"Vasometric Peak Frequency: {peak_vaso:.4f} Hz")
            
            # Compute HRV features
            self.log_output("Computing HRV features...")
            self.update_progress(90)
            
            if len(rr_intervals) > 1:
                hrv = feature(rr_intervals)
                time_features = hrv.time.get_features()
                freq_features = hrv.freq.get_features()
                nonlinear_features = hrv.non_linear.compute_all()
            else:
                time_features = {}
                freq_features = {}
                nonlinear_features = {}
                self.log_output("Warning: Not enough RR intervals for HRV analysis")
            
            # Store results
            self.results = {
                'signal_data': {
                    'hr_signal': signal_Hr,
                    'hr_peaks': peaks_Hr,
                    'resp_signal': signal_DWT[J_Resp],
                    'resp_peaks': resp_peaks,
                    'vaso_signal': signal_dwtvaso[J_vaso],
                    'original_signal': signal,
                    'fs': fs_effective
                },
                'metrics': {
                    'BPM': BPM,
                    'BrPM': BrPM,
                    'respiratory_freq_hz': peak_freq,
                    'vasometric_freq_hz': peak_vaso,
                },
                'hrv_time': time_features,
                'hrv_freq': freq_features,
                'hrv_nonlinear': nonlinear_features,
                'parameters': {
                    'original_fs': fs,
                    'effective_fs': fs_effective,
                    'factor': factor,
                    'j_resp': J_Resp,
                    'j_vaso': J_vaso
                }
            }
            
            # Update UI
            self.update_progress(100)
            self.root.after(0, self.update_results_display)
            self.root.after(0, self.create_plots)
            
            self.log_output("Analysis completed successfully!")
            
        except Exception as e:
            self.log_output(f"Error during analysis: {str(e)}")
            import traceback
            self.log_output(traceback.format_exc())
        finally:
            self.root.after(0, self.analysis_completed)
    
    def update_progress(self, value):
        """Thread-safe progress update"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def update_results_display(self):
        """Update the results treeview"""
        self.clear_results_tree()
        
        # Basic metrics
        metrics_node = self.results_tree.insert('', 'end', text='Basic Metrics', values=('', ''))
        for key, value in self.results['metrics'].items():
            if isinstance(value, float):
                display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            self.results_tree.insert(metrics_node, 'end', text=key, values=(display_value, self.get_unit(key)))
        
        # HRV Time Domain
        if self.results['hrv_time']:
            time_node = self.results_tree.insert('', 'end', text='HRV Time Domain', values=('', ''))
            for key, value in self.results['hrv_time'].items():
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                self.results_tree.insert(time_node, 'end', text=key, values=(display_value, self.get_unit(key)))
        
        # HRV Frequency Domain
        if self.results['hrv_freq']:
            freq_node = self.results_tree.insert('', 'end', text='HRV Frequency Domain', values=('', ''))
            for key, value in self.results['hrv_freq'].items():
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                self.results_tree.insert(freq_node, 'end', text=key, values=(display_value, self.get_unit(key)))
        
        # HRV Nonlinear
        if self.results['hrv_nonlinear']:
            nonlinear_node = self.results_tree.insert('', 'end', text='HRV Nonlinear', values=('', ''))
            for key, value in self.results['hrv_nonlinear'].items():
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                self.results_tree.insert(nonlinear_node, 'end', text=key, values=(display_value, self.get_unit(key)))
        
        # Expand all nodes
        for item in self.results_tree.get_children():
            self.results_tree.item(item, open=True)
    
    def get_unit(self, parameter_name):
        """Get appropriate unit for parameter"""
        unit_map = {
            'BPM': 'beats/min',
            'BrPM': 'breaths/min',
            'respiratory_freq_hz': 'Hz',
            'vasometric_freq_hz': 'Hz',
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
    
    def clear_plots(self):
        """Clear all plots"""
        self.fig.clear()
        self.canvas.draw()
    
    def create_plots(self):
        """Create enhanced signal plots with subplots"""
        self.fig.clear()
        
        if not self.results:
            return
        
        try:
            # Create subplots with more space
            gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Heart Rate Signal
            ax1 = self.fig.add_subplot(gs[0, :])
            signal_hr = self.results['signal_data']['hr_signal']
            peaks_hr = self.results['signal_data']['hr_peaks']
            fs = self.results['signal_data']['fs']
            
            # Create time axis in seconds
            time_hr = np.arange(len(signal_hr)) / fs
            
            ax1.plot(time_hr, signal_hr, label='Heart Rate Signal', color='red', linewidth=1)
            ax1.plot(time_hr[peaks_hr], signal_hr[peaks_hr], "x", label='Detected Peaks', 
                    color='blue', markersize=8, markeredgewidth=2)
            ax1.set_title(f'Heart Rate Signal - BPM: {self.results["metrics"]["BPM"]:.2f}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (seconds)', fontsize=10)
            ax1.set_ylabel('Amplitude', fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Respiratory Signal
            ax2 = self.fig.add_subplot(gs[1, 0])
            signal_resp = self.results['signal_data']['resp_signal']
            peaks_resp = self.results['signal_data']['resp_peaks']
            
            time_resp = np.arange(len(signal_resp)) / fs
            
            ax2.plot(time_resp, signal_resp, label='Respiratory Signal', color='green', linewidth=1)
            ax2.plot(time_resp[peaks_resp], signal_resp[peaks_resp], "x", label='Detected Peaks', 
                    color='orange', markersize=6, markeredgewidth=2)
            ax2.set_title(f'Respiratory Signal - Freq: {self.results["metrics"]["respiratory_freq_hz"]:.4f} Hz', 
                         fontsize=11, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=10)
            ax2.set_ylabel('Amplitude', fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Vasometric Signal
            ax3 = self.fig.add_subplot(gs[1, 1])
            signal_vaso = self.results['signal_data']['vaso_signal']
            
            time_vaso = np.arange(len(signal_vaso)) / fs
            
            ax3.plot(time_vaso, signal_vaso, label='Vasometric Signal', color='purple', linewidth=1)
            ax3.set_title(f'Vasometric Signal - Peak Freq: {self.results["metrics"]["vasometric_freq_hz"]:.4f} Hz', 
                         fontsize=11, fontweight='bold')
            ax3.set_xlabel('Time (seconds)', fontsize=10)
            ax3.set_ylabel('Amplitude', fontsize=10)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add overall title
            self.fig.suptitle('Signal Analysis Results', fontsize=14, fontweight='bold', y=0.98)
            
            # Use subplots_adjust instead of tight_layout to avoid warnings
            self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.3, hspace=0.4)
            self.canvas.draw()
            
        except Exception as e:
            self.log_output(f"Error creating plots: {str(e)}")
    
    def refresh_plots(self):
        """Refresh the plots"""
        if self.results:
            self.create_plots()
        else:
            messagebox.showwarning("Warning", "No analysis results to plot")
    
    def save_plots(self):
        """Save current plots to file"""
        if not self.results:
            messagebox.showwarning("Warning", "No plots to save")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Plots",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                messagebox.showinfo("Success", f"Plots saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plots: {str(e)}")
    
    def export_plots(self):
        """Export plots with analysis parameters as metadata"""
        if not self.results:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Analysis Plots",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            try:
                # Add metadata to the plot
                metadata_text = f"""Analysis Parameters:
Original FS: {self.results['parameters']['original_fs']} Hz
Effective FS: {self.results['parameters']['effective_fs']:.2f} Hz
Downsampling Factor: {self.results['parameters']['factor']}
Respiratory DWT Level: {self.results['parameters']['j_resp']}
Vasometric DWT Level: {self.results['parameters']['j_vaso']}

Results:
BPM: {self.results['metrics']['BPM']:.2f}
BrPM: {self.results['metrics']['BrPM']:.2f}
Respiratory Freq: {self.results['metrics']['respiratory_freq_hz']:.4f} Hz
Vasometric Freq: {self.results['metrics']['vasometric_freq_hz']:.4f} Hz"""
                
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', metadata={'Description': metadata_text})
                messagebox.showinfo("Success", f"Analysis plots exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plots: {str(e)}")
    
    def clear_results_tree(self):
        """Clear the results treeview"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
    
    def log_output(self, message):
        """Add message to output queue for thread-safe GUI updates"""
        self.output_queue.put(message + "\n")
    
    def setup_output_monitoring(self):
        """Monitor output queue and update GUI"""
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.output_text.insert(tk.END, message)
                self.output_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.setup_output_monitoring)
    
    def analysis_completed(self):
        """Called when analysis is complete"""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Analysis completed")
    
    def stop_analysis(self):
        """Stop the currently running analysis"""
        self.status_var.set("Stopping analysis...")
        self.analysis_completed()
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_var.set("Output cleared")
    
    def save_results(self):
        """Save analysis results to a file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                if filename.endswith('.json'):
                    import json
                    # Convert numpy arrays to lists for JSON serialization
                    results_copy = self.results.copy()
                    results_copy['signal_data'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                 for k, v in results_copy['signal_data'].items()}
                    
                    with open(filename, 'w') as f:
                        json.dump(results_copy, f, indent=2)
                        
                elif filename.endswith('.csv'):
                    # Create a flattened DataFrame
                    data = []
                    for category, features in [('Basic Metrics', self.results['metrics']),
                                             ('HRV Time', self.results['hrv_time']),
                                             ('HRV Frequency', self.results['hrv_freq']),
                                             ('HRV Nonlinear', self.results['hrv_nonlinear'])]:
                        for key, value in features.items():
                            data.append({
                                'Category': category,
                                'Parameter': key,
                                'Value': value,
                                'Unit': self.get_unit(key)
                            })
                    
                    df = pd.DataFrame(data)
                    df.to_csv(filename, index=False)
                    
                else:  # Text file
                    with open(filename, 'w') as f:
                        f.write("Signal Analysis Results\n")
                        f.write("=" * 50 + "\n\n")
                        
                        # Parameters
                        f.write("Analysis Parameters:\n")
                        for key, value in self.results['parameters'].items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")
                        
                        # Basic metrics
                        f.write("Basic Metrics:\n")
                        for key, value in self.results['metrics'].items():
                            unit = self.get_unit(key)
                            f.write(f"  {key}: {value} {unit}\n")
                        f.write("\n")
                        
                        # HRV features
                        if self.results['hrv_time']:
                            f.write("HRV Time Domain Features:\n")
                            for key, value in self.results['hrv_time'].items():
                                unit = self.get_unit(key)
                                f.write(f"  {key}: {value} {unit}\n")
                            f.write("\n")
                        
                        if self.results['hrv_freq']:
                            f.write("HRV Frequency Domain Features:\n")
                            for key, value in self.results['hrv_freq'].items():
                                unit = self.get_unit(key)
                                f.write(f"  {key}: {value} {unit}\n")
                            f.write("\n")
                        
                        if self.results['hrv_nonlinear']:
                            f.write("HRV Nonlinear Features:\n")
                            for key, value in self.results['hrv_nonlinear'].items():
                                unit = self.get_unit(key)
                                f.write(f"  {key}: {value} {unit}\n")
                        
                        # Analysis output
                        f.write("\nAnalysis Log:\n")
                        f.write("-" * 30 + "\n")
                        f.write(self.output_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    SignalAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()