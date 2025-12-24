import streamlit as st
from Plot import *
import Analyzer
import pandas as pd

def show(data, labels, fs, test_data, test_labels, trial_idx):
    st.title("ERD/ERS Analysis")
    data = np.concatenate([data, test_data], axis=0)
    labels = np.concatenate([labels, test_labels], axis=0)
    window_size = st.sidebar.number_input("Moving Average Window", 1, 1000, 100, 10)
    
    st.write(f"EEG Data Loaded: {data.shape[0]} trials, {data.shape[1]} channels, {data.shape[2]} samples per trial.")
    cache_key = f"erd_analyzer_{window_size}"
    
    if cache_key not in st.session_state:
        with st.spinner("Computing ERD/ERS analysis... This may take a moment."):
            data_analyzer = Analyzer.Analyzer(fs=fs)
            data_analyzer.window_size = window_size
            data_analyzer.run(data)
            
            baseline_start = int(0.0 * fs)
            baseline_end   = int(1.0 * fs)       
            
            data_analyzer.apply_baseline(baseline_start, baseline_end)
            data_analyzer.apply_spatial_filter(method='laplacian')
            
            st.session_state[cache_key] = data_analyzer
            print("ERD/ERS analysis computed and cached.")
    else:
        data_analyzer = st.session_state[cache_key]
        print("Using cached analysis results")
    
    st.subheader("ERD/ERS Data Plot")
           
    # Raw Signal
    with st.expander("🔍 Raw EEG Signal", expanded=False):
        st.write(f"Original EEG data from {trial_idx}th trial before any processing.")
        fig = plotSignal(data_analyzer.data[trial_idx:trial_idx+1], fs=fs)
        st.pyplot(fig)

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("1. BPF EEG Signal", expanded=False):
            fig = plot2Signal(data_analyzer.erd_bandpassed, data_analyzer.ers_bandpassed, 
                        fs=250, label1='ERD Band (8-11 Hz)', label2='ERS Band (26-30 Hz)', idx=trial_idx)
            st.pyplot(fig)
    with col2:
        with st.expander("2. Squared Signal", expanded=False):
            fig = plot2Signal(data_analyzer.erd_squared, data_analyzer.ers_squared, 
                        fs=250, label1='ERD Squared', label2='ERS Squared', idx=trial_idx)
            st.pyplot(fig)

    # Moving Average
    with st.expander("3. MAV EEG Signal", expanded=False):
        st.write("Moving Average over squared signal to smooth the data.")
        fig = plot2Signal(data_analyzer.erd_movingavg, data_analyzer.ers_movingavg, 
                    fs=250, label1='ERD Moving Average', label2='ERS Moving Average', idx=trial_idx)
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
                    fs=250, label1='Alpha ERD (%) - Laplacian', label2='Beta ERS (%) - Laplacian', idx=trial_idx)
        st.pyplot(fig)
    
    st.divider()
    
    extra, features = data_analyzer.motorImagery(int(1*fs), int(3*fs))
    
    st.success(f"Extracted {features.shape[1]} features from {features.shape[0]} trials")
    
    # Display feature statistics
    with st.expander("Feature Statistics", expanded=True):
        # Feature names
        feature_names = [
            # Alpha features
            'Alpha_Mean_C3', 'Alpha_Mean_Cz', 'Alpha_Mean_C4',
            'Alpha_Var_C3', 'Alpha_Var_Cz', 'Alpha_Var_C4',
            'Alpha_Std_C3', 'Alpha_Std_Cz', 'Alpha_Std_C4',
            'Alpha_Max_C3', 'Alpha_Max_Cz', 'Alpha_Max_C4',
            'Alpha_Min_C3', 'Alpha_Min_Cz', 'Alpha_Min_C4',
            # Beta features
            'Beta_Mean_C3', 'Beta_Mean_Cz', 'Beta_Mean_C4',
            'Beta_Var_C3', 'Beta_Var_Cz', 'Beta_Var_C4',
            'Beta_Std_C3', 'Beta_Std_Cz', 'Beta_Std_C4',
            'Beta_Max_C3', 'Beta_Max_Cz', 'Beta_Max_C4',
            'Beta_Min_C3', 'Beta_Min_Cz', 'Beta_Min_C4',
            # Spatial contrasts
            'Alpha_C3-C4', "Alpha_C3-C4 Ratio",
            'Beta_C3-C4', "Beta_C3-C4 Ratio"
        ]
        
        # Create DataFrame
        df_features = pd.DataFrame(features, columns=feature_names)
        df_features.insert(0, 'Trial', range(len(features)))
        df_features.insert(1, 'Class', labels)
        
        # Feature statistics by class
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Class Distribution:**")
            class_counts = pd.Series(labels).value_counts()
            st.write(f"- Class 0 (Left Hand): {class_counts.get(0, 0)} trials")
            st.write(f"- Class 1 (Right Hand): {class_counts.get(1, 0)} trials")
        
        
        with col2:
            st.write("**Feature Value Ranges:**")
            st.write(f"- Min: {features.min():.4f}")
            st.write(f"- Max: {features.max():.4f}")
            st.write(f"- Mean: {features.mean():.4f}")
            st.write(f"- Std: {features.std():.4f}")
        
        # Feature comparison between classes
        st.divider()
        cols1, cols2 = st.columns(2)
        with cols1:
            st.write("**Feature Comparison by Class:**")
            
            # Select a few important features to visualize
            important_features = ['Alpha_Mean_C3', 'Alpha_Mean_C4', 'Beta_Mean_C3', 'Beta_Mean_C4', 'Alpha_C3-C4', 'Beta_C3-C4']
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for idx, feat_name in enumerate(important_features):
                if feat_name in df_features.columns:
                    feat_idx = df_features.columns.get_loc(feat_name)
                    
                    class1_data = features[labels == 0, feat_idx - 2]  # -2 to account for Trial and Class columns
                    class2_data = features[labels == 1, feat_idx - 2]
                    
                    axes[idx].boxplot([class1_data, class2_data], labels=['Left Hand', 'Right Hand'])
                    axes[idx].set_title(feat_name)
                    axes[idx].set_ylabel('Feature Value')
                    axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with cols2:    
            # Select subset of features for correlation
            subset_features = df_features[important_features].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            import seaborn as sns
            sns.heatmap(subset_features, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            st.pyplot(fig)
        
    # Store analyzer in session state for other pages to use
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = data_analyzer
    
    return data_analyzer    