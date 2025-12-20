import numpy as np
from scipy import linalg

class CSP:
    def __init__(self, n_components=2, log=True, reg=1e-8, norm=True):
        self.n_components = n_components
        self.log = log
        self.norm_trace = norm
        self.reg = reg
        
    def _covariance_matrix(self, X):
        cov = np.zeros((X.shape[1], X.shape[1]))
        for trial in X:
            C = trial @ trial.T
            if self.norm_trace:
                C /= np.trace(C)
            cov += C
        cov /= X.shape[0]
        cov += self.reg * np.eye(cov.shape[0])
        return cov
    
    def fit(self, X, y):
        # Get unique classes and split data accordingly
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Expected 2 classes, got {len(classes)}")
        
        X1 = X[y == classes[0]]
        X2 = X[y == classes[1]]
        
        # Check if we have data for both classes
        if len(X1) == 0 or len(X2) == 0:
            raise ValueError(f"Empty class detected: class {classes[0]} has {len(X1)} samples, class {classes[1]} has {len(X2)} samples")
        
        cov1, cov2 = self._covariance_matrix(X1), self._covariance_matrix(X2)
        
        cov_sum = cov1 + cov2
        eigvals, eigvecs = linalg.eigh(cov_sum)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]
        
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
        S1 = whitening.T @ cov1 @ whitening
        
        eigvals_s, eigvecs_s = linalg.eigh(S1)
        idx = np.argsort(eigvals_s)[::-1]
        eigvecs_s = eigvecs_s[:, idx]
        
        self.filters_ = whitening @ eigvecs_s
        self.patterns_ = linalg.pinv(self.filters_)
        self.eigen_values_ = eigvals_s  # Store eigenvalues for plotting
        return self
    
    def transform(self, X):
        W = self.filters_.T
        m = self.n_components // 2
        W_sel = np.vstack([W[:m], W[-m:]])
        
        feats = np.zeros((X.shape[0], self.n_components))
        for i, trial in enumerate(X):
            Z = W_sel @ trial
            var = np.var(Z, axis=1)
            var /= np.sum(var)
            feats[i] = np.log(var + 1e-10) if self.log else var
        return feats

    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)