import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preproceed import dataLoader

train_files = ["B0101T", "B0102T", "B0103T"]

X_list, y_list = [], []
for f in train_files:
    d = dataLoader(f, data_type="train")
    X_list.append(d.X)
    y_list.append(d.y)

X_train = np.concatenate(X_list)
y_train = np.concatenate(y_list)

data_eval = dataLoader("B0103T", data_type="train")
X_test = data_eval.X

csp = CSP(csp_component=6, log=True)
X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp  = csp.transform(X_test)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train_csp, y_train)

y_pred = clf.predict(X_test_csp)

print("\nPredicted labels:")
for i, p in enumerate(y_pred):
    print(f"Trial {i+1}: {'LH' if p == 0 else 'RH'}")

print("\nSummary:")
print("LH:", np.sum(y_pred == 0))
print("RH:", np.sum(y_pred == 1))
