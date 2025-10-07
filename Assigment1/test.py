import numpy as np
import pandas as pd
import streamlit as st

from Utils import *
from Plot import *

class Coeficient:
    def __init__(self, fs=None):
        self.fs = fs
        self.original_fs = fs
        # Store only non-zero coefficients as dictionaries
        self.qj_coeffs = {}
        self.a = None
        self.b = None
    
    def getAnBvalues(self, j):
        self.a = -(2**j + 2**(j-1) - 2)
        self.b = -(1 - 2**(j-1)) + 1
        print(f"Computed a: {self.a}, b: {self.b} for j: {j}")
        return self.a, self.b
    
    def compute_qj_coefficients(self, j):
        """Compute QJ coefficients for scale j using vectorized operations"""
        a, b = self.getAnBvalues(j)
        k_range = np.arange(a, b)
        coeffs = np.zeros(len(k_range))
        
        # Define coefficient formulas for each j
        if j == 1:
            scale_factor = -2
            for i, k in enumerate(k_range):
                coeffs[i] = scale_factor * (dirac(k) - dirac(k+1))
        elif j == 2:
            scale_factor = -1/4
            for i, k in enumerate(k_range):
                coeffs[i] = scale_factor * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 
                                          2*dirac(k+2) - 3*dirac(k+3) - dirac(k+4))
        # Add other j cases as needed...
        
        return k_range, coeffs
    
    def initialize_qj_filter(self, max_j=8):
        """Initialize QJ filters up to scale max_j"""
        print("Initializing QJ filters...")
        
        plot_data = {'x': [], 'y': [], 'titles': []}
        
        for j in range(1, min(max_j + 1, 5)):  # Limit to j=4 for initial display
            k_range, coeffs = self.compute_qj_coefficients(j)
            
            # Store only non-zero coefficients
            self.qj_coeffs[j] = {
                'k_range': k_range,
                'coefficients': coeffs[coeffs != 0],
                'k_indices': k_range[coeffs != 0]
            }
            
            plot_data['x'].append(k_range)
            plot_data['y'].append(coeffs)
            plot_data['titles'].append(f"j={j}")
        
        # Plot results
        plotRow(
            x=plot_data['x'],
            y=plot_data['y'],
            plot_type="bar",
            title="QJ Filter Coefficients",
            xlabel="k",
            ylabel="Coefficient Value"
        )
        
        print(f"QJ filters initialized for j=1 to j={min(max_j, 4)}")
        
# Configure page settings first
st.set_page_config(page_title="Signal Processing App", layout="wide")

st.title("Signal Processing Application")
st.write("This application demonstrates signal processing using the Coeficient class.")

# Cache the coefficient initialization
@st.cache_resource
def initialize_coefficient(fs):
    """Cache the coefficient initialization to avoid recomputation"""
    return Coeficient(fs=fs)

# Cache data loading
@st.cache_data
def load_data(uploaded_file):
    """Cache data loading to avoid re-reading files"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        return df
    return None

# Initialize coefficient instance
coef = initialize_coefficient(fs=125)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Filter generation
    if st.button("Generate QJ Filter Plot", type="primary"):
        with st.spinner("Generating QJ filter coefficients..."):
            coef.initialize_qj_filter()
        st.success("QJ filter coefficients initialized and plotted.")
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file for signal processing", 
        type=["csv", "txt"],
        help="Upload a CSV file containing signal data"
    )
    
    # Data type selection
    data_option = st.selectbox(
        "Select data type", 
        ["RESP", "PLETH", "V", "AVR", "II"],
        help="Choose which signal column to process"
    )

# Main content area
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        
        if df is not None:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Data Info")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {', '.join(df.columns)}")
                
                if data_option in df.columns:
                    signal_data = df[data_option].dropna()
                    st.write(f"**{data_option} Stats:**")
                    st.write(f"- Length: {len(signal_data)}")
                    st.write(f"- Mean: {signal_data.mean():.2f}")
                    st.write(f"- Std: {signal_data.std():.2f}")
            
            with col1:
                st.subheader("Signal Visualization")
                
                if data_option in df.columns:
                    signal = df[data_option].dropna().values
                    time = df['Time [s]'].dropna().values[:len(signal)]
                    
                    # Use line plot for better performance with large datasets
                    plot_type = "line" if len(signal) > 1000 else "bar"
                    
                    plotSingle(
                        time, signal, 
                        plot_type=plot_type, 
                        title=f"{data_option} Signal Before Processing", 
                        xlabel="Time (s)", 
                        ylabel="Amplitude"
                    )
                else:
                    st.error(f"Column '{data_option}' not found in the uploaded data.")
                    st.write("**Available columns:**", list(df.columns))
                    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please make sure your CSV file has the correct format.")
else:
    st.info("👆 Please upload a CSV file to begin signal processing.")