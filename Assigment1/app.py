import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from Coeficient import Coeficient

st.title("Signal Processing Application")
st.write("This application demonstrates signal processing using the Coeficient class.")
st.set_page_config(layout="wide")

# Initialize Coeficient instance
fs = 125
coef = Coeficient(fs=fs)

st.write(f"Coeficient instance created with sampling rate: {fs} Hz")

# Add a button to generate the plot
if st.button("Generate QJ Filter Plot"):
    st.write("Generating QJ filter coefficients...")
    coef.initialize_qj_filter()
    st.write("QJ filter coefficients initialized and plotted.")

