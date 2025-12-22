import streamlit as st
import pandas as pd
from Utils import BPF
import CSP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show(X_train, y_train, X_test, y_test, fs, trial_idx):
    channel_names = ['C3', 'Cz', 'C4']
    n_channels = X_train.shape[1]  # Number of physical channels (3)
    
    # Maximum CSP components = 2 * number of channels
    csp_component = st.sidebar.number_input(
        "Number of CSP Components", 
        min_value=2, 
        max_value=2*n_channels,  # Can be up to 6 for 3 channels
        value=2,
        step=1
    )
    
    # ===== CACHE KEY FOR PRE-COMPUTATION =====
    cache_key = f"csp_precomputed_{csp_component}"
    
    if cache_key not in st.session_state:
        with st.spinner("🔄 Pre-computing CSP analysis for all trials... This will only happen once."):
            # Bandpass filter training data
            bandpassed = np.zeros_like(X_train)
            for t in range(X_train.shape[0]):
                for c in range(X_train.shape[1]):
                    bandpassed[t, c, :] = BPF(X_train[t, c, :], lowcut=8, highcut=30, fs=fs)
            
            # Fit CSP with user-selected number of components
            csp_analyzer = CSP.CSP(csp_component=csp_component, log=True, reg=1e-8, norm=True)
            csp_analyzer.fit(bandpassed, y_train)
            
            # Bandpass filter test data
            bandpassed_test = np.zeros_like(X_test)
            for t in range(X_test.shape[0]):
                for c in range(X_test.shape[1]):
                    bandpassed_test[t, c, :] = BPF(X_test[t, c, :], lowcut=8, highcut=30, fs=fs)
            
            # Transform all test trials
            features = csp_analyzer.transform(bandpassed_test)
            
            # Cache everything
            st.session_state[cache_key] = {
                'csp_analyzer': csp_analyzer,
                'bandpassed_test': bandpassed_test,
                'features': features,
                'X_test': X_test,
                'y_test': y_test,
                'csp_component': csp_component
            }
            st.success("✅ Pre-computation complete! All trials cached.")
    else:
        # Load from cache
        cached_data = st.session_state[cache_key]
        csp_analyzer = cached_data['csp_analyzer']
        bandpassed_test = cached_data['bandpassed_test']
        features = cached_data['features']
        X_test = cached_data['X_test']
        y_test = cached_data['y_test']
    
    # Get actual number of features (can be up to 6 for 3 channels)
    n_features = features.shape[1]
    
    # Number of spatial patterns we can actually plot (limited by number of channels = 3)
    n_spatial_components = min(csp_component, n_channels)
    
    st.title("CSP Feature Extraction")
    st.info(f"📊 Analyzing Trial #{trial_idx} from test set | Class: {'Left Hand' if y_test[trial_idx] == 0 else 'Right Hand'}")
    st.success(f"✅ Extracted {n_features} features from {features.shape[0]} trials using {n_channels} channels (cached)")
    
    if n_features > n_channels:
        st.info(f"ℹ️ CSP extracted {n_features} features from {n_channels} channels by using component pairs (most/least discriminative)")
    
    # ===== PER-TRIAL VISUALIZATION =====
    st.divider()
    st.subheader(f"🔍 Trial #{trial_idx} Analysis")
    
    with st.expander("Raw vs Bandpassed Signal (Selected Trial)", expanded=True):
        from Plot import plotSignal
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Signal:**")
            fig = plotSignal(X_test[trial_idx:trial_idx+1], fs=fs)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Bandpassed Signal (8-30 Hz):**")
            fig = plotSignal(bandpassed_test[trial_idx:trial_idx+1], fs=fs)
            st.pyplot(fig)
            plt.close()
    
    with st.expander("CSP-Filtered Signal (Selected Trial)", expanded=True):
        st.write(f"**Spatial filtering applied to Trial #{trial_idx}**")
        
        # Apply CSP filters to get source signals
        trial_data = bandpassed_test[trial_idx]  # (3, n_samples)
        
        # Get the spatial filters (limited to number of channels)
        # filters_ shape: (n_channels, n_spatial_components)
        W = csp_analyzer.filters_[:, :n_spatial_components].T  # (n_spatial_components, n_channels)
        
        # Transform: (n_spatial_components, n_samples)
        csp_sources = W @ trial_data
        
        # Plot CSP spatial components (limited by channels)
        fig, axes = plt.subplots(n_spatial_components, 1, figsize=(12, 3*n_spatial_components), sharex=True)
        if n_spatial_components == 1:
            axes = [axes]
        
        time = np.arange(trial_data.shape[1]) / fs
        
        for i in range(n_spatial_components):
            axes[i].plot(time, csp_sources[i, :], color='purple', linewidth=1)
            axes[i].set_ylabel(f'Spatial\nComponent {i+1}')
            axes[i].set_title(f'CSP Spatial Component {i+1}')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Showing {n_spatial_components} spatial components (limited by {n_channels} channels)")
    
    with st.expander("Spatial Pattern Weights (Selected Trial)", expanded=True):
        st.write(f"**How each channel contributes to spatial components**")
        
        fig, axes = plt.subplots(1, n_spatial_components, figsize=(5*n_spatial_components, 4))
        if n_spatial_components == 1:
            axes = [axes]
        
        for i in range(n_spatial_components):
            pattern = csp_analyzer.patterns_[:, i]
            filter_weights = csp_analyzer.filters_[:, i]
            
            x = np.arange(len(channel_names))
            width = 0.35
            
            axes[i].bar(x - width/2, pattern, width, label='Pattern', alpha=0.7, color='blue')
            axes[i].bar(x + width/2, filter_weights, width, label='Filter', alpha=0.7, color='red')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(channel_names)
            axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
            axes[i].set_title(f'Spatial Component {i+1}')
            axes[i].set_ylabel('Weight')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Patterns = brain sources | Filters = spatial weights")
    
    with st.expander("Feature Values: All CSP Features for Selected Trial", expanded=True):
        st.write(f"**All {n_features} CSP feature values for Trial #{trial_idx}**")
        
        # Show all feature values in a nice format
        feature_data = {
            'CSP Feature': [f'CSP {i+1}' for i in range(n_features)],
            'Log Variance': [f'{features[trial_idx, i]:.4f}' for i in range(n_features)],
            'Description': []
        }
        
        # Add descriptions for each feature
        if n_features == 2:
            feature_data['Description'] = ['Most discriminative (Class 0)', 'Most discriminative (Class 1)']
        elif n_features <= n_channels:
            for i in range(n_features):
                if i < n_features // 2:
                    feature_data['Description'].append(f'Class 0 emphasis (rank {i+1})')
                else:
                    feature_data['Description'].append(f'Class 1 emphasis (rank {n_features-i})')
        else:
            # For 4 or 6 features from 3 channels
            for i in range(n_features):
                if i < n_features // 2:
                    feature_data['Description'].append(f'Class 0 emphasis (rank {i+1})')
                else:
                    feature_data['Description'].append(f'Class 1 emphasis (rank {n_features-i})')
        
        df_features = pd.DataFrame(feature_data)
        st.dataframe(df_features, use_container_width=True)
        st.caption(f"CSP generates {n_features} features from {n_channels} channels using component pairs")
    
    with st.expander("Feature Comparison: Selected Trial vs All Trials", expanded=True):
        st.write(f"**Where does Trial #{trial_idx} fall in the distribution?**")
        
        # Use ALL features for distribution plots
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
        
        for i in range(n_features):
            # All trials distribution
            class0_features = features[y_test == 0, i]
            class1_features = features[y_test == 1, i]
            
            axes[i].hist(class0_features, bins=20, alpha=0.4, label='Left Hand (all)', color='blue')
            axes[i].hist(class1_features, bins=20, alpha=0.4, label='Right Hand (all)', color='red')
            
            # Highlight selected trial
            trial_value = features[trial_idx, i]
            trial_class = y_test[trial_idx]
            color = 'blue' if trial_class == 0 else 'red'
            
            axes[i].axvline(trial_value, color=color, linewidth=3, 
                          linestyle='--', label=f'Trial #{trial_idx}')
            axes[i].set_title(f'CSP {i+1}: Trial Value = {trial_value:.4f}')
            axes[i].set_xlabel('Log Variance')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Vertical line shows where Trial #{trial_idx} falls in the distribution for all {n_features} features")
    
    st.divider()
    
    # Display CSP components
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**CSP Filters:**")
        st.caption(f"Spatial filters that extract discriminative features (first {n_spatial_components} shown)")
        filters_df = pd.DataFrame(
            csp_analyzer.filters_[:, :n_spatial_components], 
            columns=[f'Component {i+1}' for i in range(n_spatial_components)],
            index=channel_names
        )
        st.dataframe(filters_df)
    
    with col2:
        st.write("**CSP Patterns:**")
        st.caption(f"Brain activation patterns (first {n_spatial_components} shown)")
        patterns_df = pd.DataFrame(
            csp_analyzer.patterns_[:, :n_spatial_components], 
            columns=[f'Component {i+1}' for i in range(n_spatial_components)],
            index=channel_names
        )
        st.dataframe(patterns_df)
    
    st.write("**CSP Eigenvalues:**")
    st.caption("Larger values = better class separation")
    eigenvalues_df = pd.DataFrame(
        csp_analyzer.eigen_values_, 
        columns=['Eigenvalue'],
        index=[f'Component {i+1}' for i in range(len(csp_analyzer.eigen_values_))]
    )
    st.dataframe(eigenvalues_df)
    
    st.divider()
    
    # ===== OVERALL VISUALIZATIONS (USE ALL FEATURES) =====
    st.subheader(f"📊 CSP Feature Analysis (All {n_features} Features)")
    
    with st.expander("1. Feature Distribution by Class", expanded=False):
        feature_names = [f"CSP {i+1}" for i in range(n_features)]
        
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
        
        for i in range(n_features):
            class1_features = features[y_test == 0, i]
            class2_features = features[y_test == 1, i]
            
            axes[i].hist(class1_features, bins=20, alpha=0.6, label='Left Hand', color='blue')
            axes[i].hist(class2_features, bins=20, alpha=0.6, label='Right Hand', color='red')
            axes[i].set_title(f'{feature_names[i]} Distribution')
            axes[i].set_xlabel('Log Variance')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(f"Overlapping histograms show how well each of the {n_features} CSP features separates the classes")
    
    with st.expander("2. Feature Value Ranges", expanded=False):
        feature_names = [f"CSP {i+1}" for i in range(n_features)]
        
        fig, ax = plt.subplots(figsize=(max(12, n_features*2), 6))
        
        positions = []
        data_to_plot = []
        labels_plot = []
        colors = []
        
        for i in range(n_features):
            class1_features = features[y_test == 0, i]
            class2_features = features[y_test == 1, i]
            
            positions.extend([i*3, i*3+1])
            data_to_plot.extend([class1_features, class2_features])
            labels_plot.extend([f'CSP{i+1}\nLeft', f'CSP{i+1}\nRight'])
            colors.extend(['lightblue', 'lightcoral'])
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, 
                        patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_plot, rotation=0, ha='center')
        ax.set_ylabel('Log Variance')
        ax.set_title(f'CSP Features Comparison: Left vs Right Hand (All {n_features} Features)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Box plots show median, quartiles, and outliers for each feature")
    
    if n_features >= 2:
        with st.expander("3. Feature Space Visualization (2D)", expanded=False):
            st.write("**Most Discriminative Components:**")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            class0_mask = y_test == 0
            class1_mask = y_test == 1
            
            # Use first and last features (most discriminative pair)
            ax.scatter(features[class0_mask, 0], features[class0_mask, -1], 
                      c='red', label='Left Hand (Class 0)', alpha=0.7, s=60, 
                      edgecolors='darkred', linewidths=1.5)
            ax.scatter(features[class1_mask, 0], features[class1_mask, -1], 
                      c='blue', label='Right Hand (Class 1)', alpha=0.7, s=60, 
                      edgecolors='darkblue', linewidths=1.5)
            
            ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel(f'CSP Feature 1 (Class 0 emphasis)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'CSP Feature {n_features} (Class 1 emphasis)', fontsize=12, fontweight='bold')
            ax.set_title('CSP Feature Space: Class Separation', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Good CSP: Points form two distinct clusters along diagonal. Bad CSP: Points overlap.")
            
            st.write("**Separation Quality Metrics:**")
            col1, col2 = st.columns(2)
            
            with col1:
                class0_center = np.mean(features[class0_mask], axis=0)
                class1_center = np.mean(features[class1_mask], axis=0)
                inter_class_dist = np.linalg.norm(class0_center - class1_center)
                st.metric("Inter-class Distance", f"{inter_class_dist:.3f}")
            
            with col2:
                intra_class_var = (np.mean(np.var(features[class0_mask], axis=0)) + 
                                  np.mean(np.var(features[class1_mask], axis=0))) / 2
                st.metric("Avg Intra-class Variance", f"{intra_class_var:.3f}")
    
    with st.expander("4. Feature Correlation Matrix", expanded=False):
        feature_names = [f"CSP {i+1}" for i in range(n_features)]
        features_df = pd.DataFrame(features, columns=feature_names)
        
        corr_matrix = features_df.corr()
        
        fig, ax = plt.subplots(figsize=(max(8, n_features*1.5), max(6, n_features*1.2)))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title(f'CSP Feature Correlation ({n_features} Features)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Low correlation = features capture different information")
    
    with st.expander("5. CSP Spatial Patterns (Topography)", expanded=False):
        st.write("**Brain Activation Patterns (Spatial Components):**")
        
        fig, axes = plt.subplots(1, n_spatial_components, figsize=(5*n_spatial_components, 4))
        if n_spatial_components == 1:
            axes = [axes]
        
        for i in range(n_spatial_components):
            pattern = csp_analyzer.patterns_[:, i]
            
            axes[i].bar(channel_names, pattern, color=['blue', 'green', 'red'], alpha=0.7)
            axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
            axes[i].set_title(f'Spatial Component {i+1}')
            axes[i].set_ylabel('Weight')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(f"Patterns show which channels contribute to each of {n_spatial_components} spatial components")
    
    st.subheader("📋 Feature Statistics")
    with st.expander("Detailed Statistics by Class", expanded=False):
        feature_names = [f"CSP Feature {i+1}" for i in range(n_features)]
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['Class'] = y_test
        features_df['Class'] = features_df['Class'].map({0: 'Left Hand', 1: 'Right Hand'})
        
        st.write("**Overall Statistics:**")
        st.dataframe(features_df.describe())
        
        st.write("**Statistics by Class:**")
        for class_name in ['Left Hand', 'Right Hand']:
            st.write(f"**{class_name}:**")
            class_data = features_df[features_df['Class'] == class_name].drop('Class', axis=1)
            st.dataframe(class_data.describe())
    
    with st.expander("6. Class Separability Analysis", expanded=False):
        st.write("**Feature Discriminability:**")
        
        separability = []
        for i in range(n_features):
            class1_mean = features[y_test == 0, i].mean()
            class2_mean = features[y_test == 1, i].mean()
            class1_std = features[y_test == 0, i].std()
            class2_std = features[y_test == 1, i].std()
            
            fisher_ratio = abs(class1_mean - class2_mean) / (class1_std + class2_std + 1e-10)
            separability.append(fisher_ratio)
        
        fig, ax = plt.subplots(figsize=(max(10, n_features*2), 6))
        feature_names = [f"CSP {i+1}" for i in range(n_features)]
        ax.bar(feature_names, separability, color='steelblue', alpha=0.7)
        ax.set_xlabel('CSP Feature')
        ax.set_ylabel("Fisher's Ratio (higher = better separation)")
        ax.set_title(f'Feature Discriminability (All {n_features} Features)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Fisher's ratio: (μ₁ - μ₂) / (σ₁ + σ₂)")