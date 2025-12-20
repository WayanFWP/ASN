from Utils import *
from preproceed import *
from Plot import *
from Analyzer import Analyzer
from CSP import CSP
import matplotlib.pyplot as plt
import mne

data_list = ["B0101T", "B0102T", "B0103T"]

X_all, y_all = [], []
info = None

for d in data_list:
    pre = dataLoader(d)
    X_all.append(pre.X)
    y_all.append(pre.y)
    info = pre.info  

data   = np.concatenate(X_all, axis=0)
labels = np.concatenate(y_all, axis=0)

fs = pre.fs
print("Data:", data.shape, "Labels:", labels.shape)

X_bp = np.zeros_like(data)
# Apply bandpass filter to each trial and channel
for t in range(data.shape[0]):
    for c in range(data.shape[1]):
        X_bp[t, c, :] = BPF(
            data[t, c, :])
        
csp = CSP(n_components=6)
X_csp = csp.fit_transform(data, labels)
X_test_CSP = csp.transform(data)

erp_analyzer = Analyzer(fs=fs)
erp_analyzer.run(data)
erp_analyzer.apply_baseline(int(0*fs), int(1*fs))
erp_analyzer.apply_spatial_filter("laplacian")

_, X_erp = erp_analyzer.motorImagery(int(1*fs), int(3*fs))
_ , X_test_ERP = erp_analyzer.motorImagery(int(1*fs), int(3*fs))

X = np.hstack((X_csp, X_erp))
X_test = np.hstack((X_test_CSP, X_test_ERP))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)

y_pred = LDA.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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