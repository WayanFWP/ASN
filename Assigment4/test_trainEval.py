import numpy as np
from CSP import CSP  # Use custom implementation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preproceed import dataLoader
import matplotlib.pyplot as plt
import mne

# ======================
# LOAD TRAIN DATA
# ======================
train_files = ["B0501T", "B0502T"]

X_list, y_list = [], []
info = None
for f in train_files:
    d = dataLoader(f, data_type="train")
    X_list.append(d.X)
    y_list.append(d.y)
    if info is None:
        info = d.info  # Get info for plotting

X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)

print("Train shape:", X_train.shape)
print("Train class distribution:", np.unique(y_train, return_counts=True))  
print("Train labels:", y_train[:10])  

# ======================
# LOAD TEST DATA (T FILE!)
# ======================
data_test = dataLoader("B0503T", data_type="train")
X_test = data_test.X
y_test = data_test.y

print("Test shape:", X_test.shape)

# ======================
# CSP + LDA (Using custom CSP)
# ======================
csp = CSP(csp_component=4)
X_train_csp = csp.fitTransform(X_train, y_train)
X_test_csp  = csp.transform(X_test)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train_csp, y_train)

y_pred = clf.predict(X_test_csp)

print("Test shape:", X_test.shape)
print("Test class distribution:", np.unique(y_test, return_counts=True))  
print("Test labels:", y_test[:10])  

# ======================
# EVALUATION
# ======================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["LH", "RH"]))

# ======================
# VISUALIZE CSP PATTERNS
# ======================
# 1. Topographic map using MNE
m = csp.n_components // 2  # patterns per class
selected_patterns = np.hstack([
    csp.patterns_[:, :m],      # First m patterns (class 1)
    csp.patterns_[:, -m:]      # Last m patterns (class 2)
])

# Create evoked object for topomap plotting
# Use sampling frequency from info
sfreq = info['sfreq']
evoked = mne.EvokedArray(selected_patterns, info, tmin=0, nave=1)

# Plot topomaps
fig2, axes = plt.subplots(1, csp.n_components, figsize=(12, 3))
for idx in range(csp.n_components):
    # Convert index to time value
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