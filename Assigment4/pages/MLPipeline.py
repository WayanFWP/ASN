import streamlit as st
from Plot import *
from Analyzer import Analyzer
from CSP import CSP
import pandas as pd
from Utils import BPF
from preproceed import dataLoader

import matplotlib.pyplot as plt
import numpy as np
import mne
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def bandpass(data, fs):
    data_bp = np.zeros_like(data)
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            data_bp[t, c, :] = BPF(
                data[t, c, :], lowcut=8, highcut=30, fs=fs)
    return data_bp

def loadInf():
    loader = st.sidebar.file_uploader("Upload Inference Data (.gdf)", type=["gdf"])
    
    if loader is None:
        return None
    
    data_inf = dataLoader(loader, data_type="eval")
    return data_inf.X

def show(X_train, Y_train, X_test, Y_test, fs, trial_idx):
    st.title("🤖 ML Pipeline: ERP + CSP + Classifier")
    
    # Sidebar controls
    classifier_name = st.sidebar.selectbox(
        "Select Classifier",
        ("LDA", "Random Forest", "SVM")
    )
    
    num_csp = st.sidebar.number_input("Number of CSP Components", 2, 2*X_train.shape[1], 4, 2)    
    scaler_checkbox = st.sidebar.checkbox("Apply StandardScaler", value=True)
    
    # Classifier-specific parameters
    if classifier_name == "Random Forest":
        n_estimators = st.sidebar.number_input("Number of Trees", 10, 500, 100, 10)
    elif classifier_name == "SVM":
        C = st.sidebar.number_input("SVM C Parameter", 0.01, 10.0, 1.0, 0.01)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
    
    inference = loadInf()
    
    # ===== CACHE KEY FOR PRE-COMPUTATION =====
    cache_key = f"ml_pipeline_{num_csp}_{scaler_checkbox}"
    
    if cache_key not in st.session_state:
        with st.spinner("🔄 Pre-computing features (bandpass + CSP + ERP)... This will only happen once."):
            # Bandpass filtering
            X_train_bp = bandpass(X_train, fs)
            X_test_bp  = bandpass(X_test, fs)
            
            # ========= CSP FEATURES ==========
            csp = CSP(csp_component=num_csp)
            X_train_csp = csp.fitTransform(X_train_bp, Y_train)
            X_test_csp  = csp.transform(X_test_bp)
            
            # ========= ERP FEATURES - TRAINING SET ==========
            erp_train = Analyzer(fs=fs)
            erp_train.run(X_train, remove_erp=False)
            erp_train.apply_baseline(int(0*fs), int(1*fs))
            erp_train.apply_spatial_filter("laplacian")
            X_train_erp, _ = erp_train.motorImagery(int(1*fs), int(3*fs))
            
            # ========= ERP FEATURES - TEST SET ==========
            erp_test = Analyzer(fs=fs)
            erp_test.run(X_test, remove_erp=False)
            erp_test.apply_baseline(int(0*fs), int(1*fs))
            erp_test.apply_spatial_filter("laplacian")
            X_test_erp, _ = erp_test.motorImagery(int(1*fs), int(3*fs))
            
            # Combine features
            X_train_combined = np.hstack((X_train_csp, X_train_erp))
            X_test_combined  = np.hstack((X_test_csp, X_test_erp))
            
            # Scale features
            scaler = None
            if scaler_checkbox:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_combined)
                X_test_scaled  = scaler.transform(X_test_combined)
            else:
                X_train_scaled = X_train_combined
                X_test_scaled = X_test_combined
            
            # Cache everything
            st.session_state[cache_key] = {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'X_train_combined': X_train_combined,
                'X_test_combined': X_test_combined,
                'n_csp_features': X_train_csp.shape[1],
                'n_erp_features': X_train_erp.shape[1],
                'scaler': scaler,
                'csp': csp,
                'X_train_bp': X_train_bp,
                'X_test_bp': X_test_bp
            }
            st.success("✅ Features cached! Switching models will now be instant.")
    
    # Load from cache
    cached_data = st.session_state[cache_key]
    X_train_scaled = cached_data['X_train_scaled']
    X_test_scaled = cached_data['X_test_scaled']
    X_train_combined = cached_data['X_train_combined']
    X_test_combined = cached_data['X_test_combined']
    n_csp_features = cached_data['n_csp_features']
    n_erp_features = cached_data['n_erp_features']
    scaler = cached_data['scaler']
    csp = cached_data['csp']
    X_train_bp = cached_data['X_train_bp']
    X_test_bp = cached_data['X_test_bp']
    
    # Feature names
    feature_names = [f'CSP_{i+1}' for i in range(n_csp_features)] + [f'ERP_{i+1}' for i in range(n_erp_features)]
    
    # Train classifier (this is fast)
    if classifier_name == "LDA":
        classifier = LinearDiscriminantAnalysis()
    elif classifier_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:  # SVM
        classifier = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    
    classifier.fit(X_train_scaled, Y_train)
    Y_pred = classifier.predict(X_test_scaled)
    Y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    # ========== DISPLAY RESULTS ==========
    st.divider()
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    acc = accuracy_score(Y_test, Y_pred)
    
    with col1:
        st.metric("Test Accuracy", f"{acc*100:.2f}%")
    with col2:
        st.metric("Total Features", len(feature_names))
    with col3:
        st.metric("CSP Components", num_csp)
    
    # Confusion Matrix
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(Y_test, Y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Left Hand', 'Right Hand'],
                   yticklabels=['Left Hand', 'Right Hand'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{classifier_name} Confusion Matrix')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Classification Report**")
        report = classification_report(Y_test, Y_pred, target_names=['Left Hand', 'Right Hand'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    # ========== CLASS DISTRIBUTION VISUALIZATION ==========
    st.divider()
    st.subheader("📊 Class Distribution Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Test Set - Actual Labels**")
        fig, ax = plt.subplots(figsize=(6, 5))
        actual_counts = np.bincount(Y_test)
        bars = ax.bar(['Left Hand', 'Right Hand'], actual_counts, 
                     color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Actual Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Total: {len(Y_test)} samples")
    
    with col2:
        st.write("**Test Set - Predictions**")
        fig, ax = plt.subplots(figsize=(6, 5))
        pred_counts = np.bincount(Y_pred)
        bars = ax.bar(['Left Hand', 'Right Hand'], pred_counts, 
                     color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Predicted Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Total: {len(Y_pred)} predictions")
    
    with col3:
        st.write("**Prediction vs Actual Comparison**")
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual_counts, width, label='Actual', 
                      color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', 
                      color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Count')
        ax.set_title('Actual vs Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(['Left Hand', 'Right Hand'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Calculate distribution shift
        distribution_diff = pred_counts - actual_counts
        st.caption(f"Difference: LH {distribution_diff[0]:+d}, RH {distribution_diff[1]:+d}")
    
    # Detailed statistics
    with st.expander("📈 Detailed Distribution Statistics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Actual Distribution:**")
            actual_dist_df = pd.DataFrame({
                'Class': ['Left Hand', 'Right Hand'],
                'Count': actual_counts,
                'Percentage': [f"{(c/len(Y_test)*100):.2f}%" for c in actual_counts]
            })
            st.dataframe(actual_dist_df, use_container_width=True)
        
        with col2:
            st.write("**Predicted Distribution:**")
            pred_dist_df = pd.DataFrame({
                'Class': ['Left Hand', 'Right Hand'],
                'Count': pred_counts,
                'Percentage': [f"{(c/len(Y_pred)*100):.2f}%" for c in pred_counts]
            })
            st.dataframe(pred_dist_df, use_container_width=True)
        
        # Prediction accuracy per class
        st.write("**Per-Class Accuracy:**")
        class_acc_df = pd.DataFrame({
            'Class': ['Left Hand', 'Right Hand'],
            'Correct Predictions': [cm[0,0], cm[1,1]],
            'Total Actual': actual_counts,
            'Accuracy': [f"{(cm[0,0]/actual_counts[0]*100):.2f}%", 
                        f"{(cm[1,1]/actual_counts[1]*100):.2f}%"]
        })
        st.dataframe(class_acc_df, use_container_width=True)

    # Feature Importance
    st.divider()
    st.subheader("🎯 Feature Importance")
    
    if classifier_name == "LDA" or (classifier_name == "SVM" and kernel == "linear"):
        # Get coefficients
        if classifier_name == "LDA":
            importance = np.abs(classifier.coef_[0])
        else:
            importance = np.abs(classifier.coef_[0])
        
        top_5_idx = np.argsort(importance)[-5:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        indices = np.argsort(importance)[::-1]
        ax.bar(range(len(importance)), importance[indices], color='steelblue')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Absolute Coefficient')
        ax.set_title(f'{classifier_name} Feature Importance')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.write("**Top 5 Most Important Features:**")
        top_features_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_5_idx],
            'Importance': [importance[i] for i in top_5_idx]
        })
        st.dataframe(top_features_df, use_container_width=True)
        
    elif classifier_name == "Random Forest":
        importance = classifier.feature_importances_
        top_5_idx = np.argsort(importance)[-5:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        indices = np.argsort(importance)[::-1]
        ax.bar(range(len(importance)), importance[indices], color='forestgreen')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Random Forest Feature Importance')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.write("**Top 5 Most Important Features:**")
        top_features_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_5_idx],
            'Importance': [importance[i] for i in top_5_idx]
        })
        st.dataframe(top_features_df, use_container_width=True)
    else:
        st.info("Feature importance not available for non-linear SVM kernel")
    
    # ========== PER-TRIAL ANALYSIS ==========
    st.divider()
    st.subheader(f"🔍 Detailed Analysis: Trial #{trial_idx}")
    
    actual_label = "Left Hand" if Y_test[trial_idx] == 0 else "Right Hand"
    pred_label = "Left Hand" if Y_pred[trial_idx] == 0 else "Right Hand"
    confidence = Y_pred_proba[trial_idx][Y_pred[trial_idx]] * 100
    is_correct = Y_test[trial_idx] == Y_pred[trial_idx]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Actual Class", actual_label)
    with col2:
        st.metric("Predicted Class", pred_label, 
                 delta="✓ Correct" if is_correct else "✗ Wrong",
                 delta_color="normal" if is_correct else "inverse")
    with col3:
        st.metric("Confidence", f"{confidence:.2f}%")
    with col4:
        prob_lh = Y_pred_proba[trial_idx][0] * 100
        prob_rh = Y_pred_proba[trial_idx][1] * 100
        st.write(f"**Probabilities:**")
        st.write(f"LH: {prob_lh:.1f}%")
        st.write(f"RH: {prob_rh:.1f}%")
    
    # Feature contribution for selected trial
    with st.expander("Feature Contributions for Selected Trial", expanded=True):
        trial_features = X_test_scaled[trial_idx]
        
        if classifier_name == "LDA" or (classifier_name == "SVM" and kernel == "linear"):
            if classifier_name == "LDA":
                w = classifier.coef_[0]
                b = classifier.intercept_[0]
            else:
                w = classifier.coef_[0]
                b = classifier.intercept_[0]
            
            decision_value = np.dot(w, trial_features) + b
            contributions = w * trial_features
            
            st.write(f"**Decision Value:** {decision_value:.4f} → {'LH' if decision_value < 0 else 'RH'}")
            st.write(f"**Distance to Decision Boundary:** {abs(decision_value):.4f}")
            
            # Plot contributions
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['red' if c < 0 else 'blue' for c in contributions]
            ax.bar(range(len(contributions)), contributions, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xticks(range(len(contributions)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_xlabel('Features')
            ax.set_ylabel('Contribution to Decision')
            ax.set_title(f'Feature Contributions for Trial #{trial_idx} (Red=LH, Blue=RH)')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Top contributing features
            top_contrib_idx = np.argsort(np.abs(contributions))[-5:][::-1]
            contrib_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_contrib_idx],
                'Contribution': [contributions[i] for i in top_contrib_idx],
                'Direction': ['→ LH' if contributions[i] < 0 else '→ RH' for i in top_contrib_idx]
            })
            st.write("**Top 5 Contributing Features:**")
            st.dataframe(contrib_df, use_container_width=True)
            
        elif classifier_name == "Random Forest":
            importance = classifier.feature_importances_
            contributions = importance * trial_features
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(contributions)), contributions, color='forestgreen', alpha=0.7)
            ax.set_xticks(range(len(contributions)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_xlabel('Features')
            ax.set_ylabel('Weighted Feature Value')
            ax.set_title(f'Feature Contributions for Trial #{trial_idx}')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            top_contrib_idx = np.argsort(np.abs(contributions))[-5:][::-1]
            contrib_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_contrib_idx],
                'Contribution': [contributions[i] for i in top_contrib_idx],
                'Importance': [importance[i] for i in top_contrib_idx]
            })
            st.write("**Top 5 Contributing Features:**")
            st.dataframe(contrib_df, use_container_width=True)
        else:
            st.info("Feature contribution visualization not available for non-linear SVM")
    
    # All trials accuracy visualization
    with st.expander("Per-Trial Prediction Results", expanded=False):
        st.write("**Prediction accuracy for each test trial:**")
        
        results_df = pd.DataFrame({
            'Trial': range(1, len(Y_test)+1),
            'Actual': ['LH' if y == 0 else 'RH' for y in Y_test],
            'Predicted': ['LH' if y == 0 else 'RH' for y in Y_pred],
            'Correct': Y_test == Y_pred,
            'Confidence (%)': [Y_pred_proba[i][Y_pred[i]] * 100 for i in range(len(Y_pred))],
            'LH Prob (%)': [Y_pred_proba[i][0] * 100 for i in range(len(Y_pred))],
            'RH Prob (%)': [Y_pred_proba[i][1] * 100 for i in range(len(Y_pred))]
        })
                        
        # Highlight selected trial
        def highlight_selected(row):
            if row.name == trial_idx:
                return ['background-color: yellow'] * len(row)
            elif row['Correct']:
                return ['background-color: lightgreen'] * len(row)
            else:
                return ['background-color: lightcoral'] * len(row)
        
        st.dataframe(results_df.style.apply(highlight_selected, axis=1), use_container_width=True)
        st.caption("🟨 Selected Trial | 🟩 Correct Prediction | 🟥 Wrong Prediction")
        
        # Confidence distribution
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['green' if c else 'red' for c in (Y_test == Y_pred)]
        ax.bar(range(len(Y_pred)), [Y_pred_proba[i][Y_pred[i]] * 100 for i in range(len(Y_pred))], 
               color=colors, alpha=0.6)
        ax.axvline(x=trial_idx, color='orange', linestyle='--', linewidth=2, label=f'Selected Trial #{trial_idx}')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Prediction Confidence for All Test Trials')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # ========== INFERENCE SECTION ==========
    if inference is not None:
        st.divider()
        st.subheader("🔮 Inference Results")
        
        inference_bp = bandpass(inference, fs)
        inference_csp = csp.transform(inference_bp)
        
        erp_inf = Analyzer(fs=fs)
        erp_inf.run(inference, remove_erp=False)
        erp_inf.apply_baseline(int(0*fs), int(1*fs))
        erp_inf.apply_spatial_filter("laplacian")
        X_inf_erp, _ = erp_inf.motorImagery(int(1*fs), int(3*fs))
        
        X_inference = np.hstack((inference_csp, X_inf_erp))
        
        if scaler_checkbox and scaler is not None:
            X_inference = scaler.transform(X_inference)
        
        y_inference = classifier.predict(X_inference)
        y_inference_proba = classifier.predict_proba(X_inference)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Inference Samples", len(y_inference))
        with col2:
            st.metric("Left Hand Predictions", np.sum(y_inference == 0))
        with col3:
            st.metric("Right Hand Predictions", np.sum(y_inference == 1))
        
        # Inference predictions table
        pred_df = pd.DataFrame({
            'Trial': range(1, len(y_inference)+1),
            'Prediction': ['Left Hand' if p == 0 else 'Right Hand' for p in y_inference],
            'Confidence (%)': [y_inference_proba[i][y_inference[i]] * 100 for i in range(len(y_inference))],
            'LH Prob (%)': [y_inference_proba[i][0] * 100 for i in range(len(y_inference))],
            'RH Prob (%)': [y_inference_proba[i][1] * 100 for i in range(len(y_inference))]
        })
        st.dataframe(pred_df, use_container_width=True)
        
        # Inference confidence plot
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['blue' if p == 0 else 'red' for p in y_inference]
        ax.bar(range(len(y_inference)), [y_inference_proba[i][y_inference[i]] * 100 for i in range(len(y_inference))],
               color=colors, alpha=0.6)
        ax.set_xlabel('Inference Trial')
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Inference Prediction Confidence (Blue=LH, Red=RH)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("📤 Upload inference data (.gdf) in the sidebar to see predictions")