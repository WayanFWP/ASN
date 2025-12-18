import streamlit as st
from preproceed import dataLoader
from Utils import *
from Plot import *
import Analyzer

class GUI:
    def __init__(self):
        # initialize
        st.set_page_config(page_title="EEG Signal Analysis GUI", layout="wide")
        self.loader = st.sidebar.file_uploader("Upload EEG Data File", type=["gdf"]) 
        if self.loader is not None:
            preproced = dataLoader(self.loader)
            self.data, self.labels = preproced.X, preproced.y
            self.fs = preproced.fs
            self.trial_idx = st.sidebar.number_input(
                "Select Trial Index",
                min_value=0,
                max_value=max(self.data.shape[0] - 1, 0),
                value=0,
                step=1
            )
            self.window_size = st.sidebar.number_input(
                "Moving Average Window Size (samples)",
                min_value=1,
                max_value=1000,
                value=100,
                step=1
            )
        else:
            st.error("Please upload a .gdf file to begin analysis.")
        
    def run(self):
        if self.loader is None:
            st.title("EEG Signal Analysis GUI")
            st.write("This GUI allows you to visualize EEG signal processing results.")

            # Sidebar for user inputs
            st.sidebar.header("User Inputs")
        else:
            self.pipeline()

    def pipeline(self):
        data_analyzer = Analyzer.Analyzer(fs=self.fs)
        data_analyzer.window_size = self.window_size
        data_analyzer.run(self.data)
        
        baseline_start = int(0.0 * self.fs )
        baseline_end   = int(1.0 * self.fs )       
        
        data_analyzer.apply_baseline(baseline_start, baseline_end)
        data_analyzer.apply_spatial_filter(method='laplacian') 
                
        st.subheader("ERD/ERS Data Plot")
                
        # Raw Signal
        with st.expander("🔍 Raw EEG Signal", expanded=False):
            st.write(f"Original EEG data from {self.trial_idx}th trial before any processing.")
            fig = plotSignal(data_analyzer.data, fs=self.fs)
            st.pyplot(fig)

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("1. BPF EEG Signal", expanded=False):
                fig = plot2Signal(data_analyzer.erd_bandpassed, data_analyzer.ers_bandpassed, 
                            fs=250, label1='ERD Band (8-11 Hz)', label2='ERS Band (26-30 Hz)', idx=self.trial_idx)
                st.pyplot(fig)
        with col2:
            with st.expander("2. Squared Signal", expanded=False):
                fig = plot2Signal(data_analyzer.erd_squared, data_analyzer.ers_squared, 
                            fs=250, label1='ERD Squared', label2='ERS Squared', idx=self.trial_idx)
                st.pyplot(fig)

        # Moving Average
        with st.expander("3. MAV EEG Signal", expanded=False):
            st.write("Moving Average over squared signal to smooth the data.")
            fig = plot2Signal(data_analyzer.erd_movingavg, data_analyzer.ers_movingavg, 
                        fs=250, label1='ERD Moving Average', label2='ERS Moving Average', idx=self.trial_idx)
            st.pyplot(fig)

        st.divider()

        # Final ERD/ERS
        with st.expander("4. Baseline Norm using laplacian", expanded=True):
            st.caption("Final ERD/ERS values after spatial filtering and baseline normalization")
            col1, col2 = st.columns(2)
            with col1:
                st.info("**ERD (Alpha)**: Event-Related Desynchronization")
            with col2:
                st.success("**ERS (Beta)**: Event-Related Synchronization")
                
            fig = merge2Signals(data_analyzer.alpha, data_analyzer.beta,
                        fs=250, label1='Alpha ERD (%) - Laplacian', label2='Beta ERS (%) - Laplacian', idx=self.trial_idx)
            st.pyplot(fig)
        
def main():
    gui = GUI()
    gui.run()

if __name__ == "__main__":
    main()