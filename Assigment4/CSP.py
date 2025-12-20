import numpy as np
from Utils import *
from scipy import linalg

class CSP:
    def __init__(self, n=2, log=True, reg=None, norm=True):
        self.n = n  # number of spatial filters per class
        self.log = log
        self.norm_trace = norm
        
        self.filters_ = None
        self.patterns_ = None
        
    def _covariance_matrix(self, X):
        # X shape: (n_trials, n_channels, n_times)
        # Initialize covariance matrix
        n_trials, n_channels, n_samples = X.shape
        
        cov = np.zeros((n_channels, n_channels))
        
        # Compute covariance for each trial and average
        for trial in range(n_trials):
            # X[trial] shape: (n_channels, n_samples)
            trial_cov = np.dot(X[trial], X[trial].T)
            
            # Normalize by trace if specified
            if self.norm_trace:
                trial_cov = trial_cov / np.trace(trial_cov)
            
            cov += trial_cov
        
        # Average across trials
        cov = cov / n_trials
        
        # Apply regularization if specified
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)
        
        return cov
    
    def composite(self, cov_class1, cov_class2):
        composite_cov = cov_class1 + cov_class2
        return composite_cov
    
    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError(f"invalid input array must be 3D, got current shape {X.shape}")
        
        # X shape: (n_trials, n_channels, n_times)
        # y shape: (n_trials,)
        class_labels = np.unique(y)
        if len(class_labels) != 2:
            raise ValueError("CSP implementation supports only two classes.")
        
        X1 = X[y == class_labels[0]]
        X2 = X[y == class_labels[1]]
        
        cov1 = self._covariance_matrix(X1)
        cov2 = self._covariance_matrix(X2)
        
        cov = self.composite(cov1, cov2)
        
        # Eigenvalue decomposition of composite covariance
        EVal_comp, EVec_comp = linalg.eigh(cov)
        
        ix = np.argsort(EVal_comp)[::-1]
        EVal_comp = EVal_comp[ix]
        EVec_comp = EVec_comp[:, ix]
        
        # P = Lambda^(-1/2) * U^T
        whitening_matrix = np.dot(
            np.diag(np.sqrt(1.0 / EVal_comp)),
            EVec_comp.T
        )
        
        # Transform covariance matrix of class 1
        S_1 = np.dott(np.dot(whitening_matrix, cov1), whitening_matrix.T)
        
        # Eigenvalue decomposition of transformed covariance
        eigen_values, eigen_vectors = linalg.eigh(S_1)
        
        # Sort by eigenvalues in descending order
        ix = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[ix]
        eigen_vectors = eigen_vectors[:, ix]
        
        # CSP projection matrix (spatial filters)
        self.filters_ = np.dot(eigen_vectors.T, whitening_matrix)
        self.eigen_values_ = eigen_values
        
        # Compute spatial patterns (inverse of filters)
        self.patterns_ = linalg.pinv(self.filters_)
        
        return self
    
    def transform(self, X):
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before Transform.\n\033 Hint: Call fit()\033")
        
        n_trial, n_channel, sample = X.shape
        n_filters_per_side = self.n_components // 2
        selected_filters = np.concatenate([
            self.filters_[:n_filters_per_side],
            self.filters_[-n_filters_per_side:]
        ])
        
        # feature matrix
        feature = np.zero((n_trial, self.n_components))
        
        for trial in range (n_trial):
            # Z = W * X, where W is spatial filter
            projected = np.dot(selected_filters, X[trial])
            
            variance = np.var(projected, axis=1)
            variance = variance / np.sum(variance)
            
            if self.log:
                feature[trial] = np.log(variance)
            else: 
                feature[trial] = variance
                
        return feature
            
    def fitTransform(self, X, y):
        return self.fit(X, y).transform(X)
    
    def getFilter(self):
        return self.filters_
    
    def getPattern(self):
        return self.patterns_    
    
    