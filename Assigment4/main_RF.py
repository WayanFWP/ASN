from Utils import *
from preproceed import *
from Plot import *
from Analyzer import Analyzer
from CSP import CSP
import matplotlib.pyplot as plt
import mne
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

train_subjects = ["B0101T", "B0102T",
                  "B0201T", "B0202T",
                  "B0301T", "B0302T",
                  "B0401T", "B0402T",
                  "B0501T", "B0502T",
                  "B0601T", "B0602T",
                  "B0701T", "B0702T",
                  "B0801T", "B0802T",
                  "B0901T", "B0902T"]

test_subjects = ["B0103T", "B0203T", "B0303T", 
                 "B0403T", "B0503T", "B0603T", 
                 "B0703T", "B0803T", "B0903T"]

inference = "B0304E"  

# Load inference data
inf_pre = dataLoader(inference, data_type="eval")
X_inf, _ = inf_pre.X, inf_pre.y
print("Inference Data:", X_inf.shape)

# Load training data
X_train_all, y_train_all = [], []
info = None

for subject in train_subjects:
    pre = dataLoader(subject)
    X_train_all.append(pre.X)
    y_train_all.append(pre.y)
    info = pre.info

X_train_raw = np.concatenate(X_train_all, axis=0)
y_train = np.concatenate(y_train_all, axis=0)

# Load validation/test data
X_test_all, y_test_all = [], []

for subject in test_subjects:
    pre = dataLoader(subject)
    X_test_all.append(pre.X)
    y_test_all.append(pre.y)

X_test_raw = np.concatenate(X_test_all, axis=0)
y_test = np.concatenate(y_test_all, axis=0)

fs = pre.fs
print(f"Training data: {X_train_raw.shape}, Labels: {y_train.shape}")
print(f"Test data: {X_test_raw.shape}, Labels: {y_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Bandpass filter for training data
X_train_raw_bp = np.zeros_like(X_train_raw)
for t in range(X_train_raw.shape[0]):
    for c in range(X_train_raw.shape[1]):
        X_train_raw_bp[t, c, :] = BPF(
            X_train_raw[t, c, :],
            lowcut=8,
            highcut=30,
            fs=fs
        )

# Bandpass filter for test data
X_test_raw_bp = np.zeros_like(X_test_raw)
for t in range(X_test_raw.shape[0]):
    for c in range(X_test_raw.shape[1]):
        X_test_raw_bp[t, c, :] = BPF(
            X_test_raw[t, c, :],
            lowcut=8,
            highcut=30,
            fs=fs
        )

# ========== CSP FEATURES ==========
csp = CSP(n_components=4)
X_train_csp = csp.fitTransform(X_train_raw_bp, y_train)
X_test_csp = csp.transform(X_test_raw_bp)

# ========== ERP FEATURES - TRAINING SET ==========
erp_train = Analyzer(fs=fs)
erp_train.run(X_train_raw, remove_erp=False)
erp_train.apply_baseline(int(0*fs), int(1*fs))
erp_train.apply_spatial_filter("laplacian")
X_train_erp, _ = erp_train.motorImagery(int(1*fs), int(3*fs))

# ========== ERP FEATURES - TEST SET ==========
erp_test = Analyzer(fs=fs)
erp_test.run(X_test_raw, remove_erp=False)
erp_test.apply_baseline(int(0*fs), int(1*fs))
erp_test.apply_spatial_filter("laplacian")
X_test_erp, _ = erp_test.motorImagery(int(1*fs), int(3*fs))

print(f"ERP Features - Train: {X_train_erp.shape}, Test: {X_test_erp.shape}")
print(f"CSP Features - Train: {X_train_csp.shape}, Test: {X_test_csp.shape}")

# ========= Inference feature ==========
X_inf_bp = np.zeros_like(X_inf)
for t in range(X_inf.shape[0]):
    for c in range(X_inf.shape[1]):
        X_inf_bp[t, c, :] = BPF(
            X_inf[t, c, :],
            lowcut=8,
            highcut=30,
            fs=fs
        )
X_inf_csp = csp.transform(X_inf_bp)
erp_inf = Analyzer(fs=fs)
erp_inf.run(X_inf, remove_erp=False)
erp_inf.apply_baseline(int(0*fs), int(1*fs))
erp_inf.apply_spatial_filter("laplacian")
X_inf_erp, _ = erp_inf.motorImagery(int(1*fs), int(3*fs))

# Combine features
X_train = np.hstack((X_train_csp, X_train_erp))
X_test = np.hstack((X_test_csp, X_test_erp))
X_inf_features = np.hstack((X_inf_csp, X_inf_erp))

print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")
print(f"Inference features: {X_inf_features.shape}")

# ============= Train and Evaluate Random Forest Classifier =============
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_inf_features = scaler.transform(X_inf_features)

RF = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)
y_inf = RF.predict(X_inf_features)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

n_csp = X_train_csp.shape[1]
n_erp = X_train_erp.shape[1]
feature_names = [f'CSP_{i+1}' for i in range(n_csp)] + [f'ERP_{i+1}' for i in range(n_erp)]

# ========== XAI FOR RANDOM FOREST ==========
print("\n" + "="*50)
print("TOP 5 MOST IMPORTANT FEATURES (Random Forest):")
print("="*50)

rf_importance = RF.feature_importances_
top_features_idx = np.argsort(rf_importance)[-5:][::-1]

for idx in top_features_idx:
    print(f"{feature_names[idx]}: {rf_importance[idx]:.4f}")

# ========== RANDOM FOREST INFERENCE EXPLANATION ==========
print("\n" + "="*50)
print("RANDOM FOREST INFERENCE EXPLANATION")
print("="*50)

for i in range(X_inf_features.shape[0]):
    x = X_inf_features[i]
    
    pred = y_inf[i]
    prob = RF.predict_proba(x.reshape(1, -1))[0]
    confidence = prob[pred] * 100
    
    pred_label = "LH (Left Hand)" if pred == 0 else "RH (Right Hand)"
    
    print(f"\n--- Trial {i+1} ---")
    print(f"Prediction      : {pred_label}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"Probability [LH, RH]: [{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Feature contribution (using feature importance * feature value)
    contrib = rf_importance * x
    top_idx = np.argsort(np.abs(contrib))[-3:][::-1]
    
    print("Top contributing features:")
    for idx in top_idx:
        sign = "-> LH" if contrib[idx] < 0 else "-> RH"
        print(f"  - {feature_names[idx]}: {contrib[idx]:+.4f} {sign} (importance: {rf_importance[idx]:.4f})")

# ========== Feature Importance Visualization ==========
plt.figure(figsize=(12, 6))
indices = np.argsort(rf_importance)[-10:][::-1]
plt.bar(range(len(indices)), rf_importance[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()

print("\nInference Summary:")
print("LH:", np.sum(y_inf == 0))
print("RH:", np.sum(y_inf == 1))

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")

# Calculate mean probability confidence for inference predictions
mean_confidence_lh = np.mean([RF.predict_proba(X_inf_features[i:i+1])[0][0] * 100 
                               for i in range(len(y_inf)) if y_inf[i] == 0]) if np.sum(y_inf == 0) > 0 else 0
mean_confidence_rh = np.mean([RF.predict_proba(X_inf_features[i:i+1])[0][1] * 100 
                               for i in range(len(y_inf)) if y_inf[i] == 1]) if np.sum(y_inf == 1) > 0 else 0

print(f"\nInference Predictions:")
print(f"  - Left Hand (LH):  {np.sum(y_inf == 0)} predictions (avg confidence: {mean_confidence_lh:.2f}%)")
print(f"  - Right Hand (RH): {np.sum(y_inf == 1)} predictions (avg confidence: {mean_confidence_rh:.2f}%)")

all_probs = RF.predict_proba(X_inf_features)
overall_confidence = np.mean([all_probs[i][y_inf[i]] * 100 for i in range(len(y_inf))])
max_confidence = np.max([all_probs[i][y_inf[i]] * 100 for i in range(len(y_inf))])
under_confidence = np.min([all_probs[i][y_inf[i]] * 100 for i in range(len(y_inf))])
print(f"  - Overall Mean Confidence: {overall_confidence:.2f}%")
print(f"  - Overall Max Confidence: {max_confidence:.2f}%")
print(f"  - Overall Under Confidence: {under_confidence:.2f}%")

m = csp.n_components // 2
selected_patterns = np.hstack([
    csp.patterns_[:, :m],
    csp.patterns_[:, -m:]
])

sfreq = info['sfreq']
evoked = mne.EvokedArray(selected_patterns, info, tmin=0, nave=1)

fig2, axes = plt.subplots(1, csp.n_components, figsize=(12, 3))
for idx in range(csp.n_components):
    time_point = idx / sfreq
    evoked.plot_topomap(
        times=[time_point], 
        axes=axes[idx] if csp.n_components > 1 else axes,
        show=False,
        colorbar=False,
        time_format=f'CSP Pattern {idx+1}'
    )

plt.suptitle('CSP Spatial Patterns (Topographic Maps)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()