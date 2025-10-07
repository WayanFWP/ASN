import numpy as np
import streamlit as st
import pandas as pd

from Coeficient import Coeficient
from Plot import *

st.title("Signal Processing Application")
st.write("This application demonstrates signal processing using the Coeficient class.")
st.set_page_config(layout="wide")

# Initialize Coeficient instance
coef = Coeficient(fs=125)

if st.button("Generate QJ Filter Plot"):
    st.write("Generating QJ filter coefficients...")
    coef.initialize_qj_filter()
    st.write("QJ filter coefficients initialized and plotted.")

data = st.sidebar.file_uploader("Upload CSV file for signal processing", type=["csv", "txt"])
# data = "data/bidmc_01_Signals.csv"

data_option = st.sidebar.selectbox("Select data type", ["RESP", "PLETH", "V", "AVR", "II"])

if data is not None:
    try:
        df = pd.read_csv(data)

        if df is not None:
            df.columns = df.columns.str.strip() 
            
            st.write("Data Before Processing:")
            if data_option in df.columns:
                signal = df[data_option].dropna().values
                time = df['Time [s]'].dropna().values 
                plotSingle(time, signal, plot_type="bar", title=f"{data_option} Signal Before Processing", xlabel="Time (s)", ylabel="Amplitude")

            else:
                st.error(f"Column '{data_option}' not found in the uploaded data.")
    except Exception as e:
        st.error(f"Error loading data: {e}")