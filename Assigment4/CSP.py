import numpy as np
from scipy import linalg

class CSP:
    def __init__(self, n_components=4, log=True, reg=None, norm=True):
        self.n_components = n_components  # total components (e.g., 4 = 2 per class)
        self.log = log
        self.norm_trace = norm
        self.reg = reg
        
        self.filters_ = None
        self.patterns_ = None
        self.eigen_values_ = None
        
    def _covariance_matrix(self, X):
        n_trials, n_channels, n_samples = X.shape
        cov = np.zeros((n_channels, n_channels))
        
        for trial in range(n_trials):
            trial_cov = np.dot(X[trial], X[trial].T)
            
            if self.norm_trace:
                trial_cov = trial_cov / np.trace(trial_cov)
            
            cov += trial_cov
        
        cov = cov / n_trials
        
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)
        
        return cov
    
    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError(f"Invalid input: must be 3D, got shape {X.shape}")
        
        class_labels = np.unique(y)
        if len(class_labels) != 2:
            raise ValueError("CSP supports only 2 classes")
        
        X1 = X[y == class_labels[0]]
        X2 = X[y == class_labels[1]]
        
        cov1 = self._covariance_matrix(X1)
        cov2 = self._covariance_matrix(X2)
        
        # Composite covariance
        cov_composite = cov1 + cov2
        
        # Eigendecomposition of composite
        EVal_comp, EVec_comp = linalg.eigh(cov_composite)
        
        # Sort descending
        ix = np.argsort(EVal_comp)[::-1]
        EVal_comp = EVal_comp[ix]
        EVec_comp = EVec_comp[:, ix]
        
        # Whitening matrix P = Λ^(-1/2) * U^T
        whitening = np.dot(np.diag(1.0 / np.sqrt(EVal_comp)), EVec_comp.T)
        
        # Transform cov1
        S1 = np.dot(np.dot(whitening, cov1), whitening.T)
        
        # Eigen decomposition of S1
        eigen_values, eigen_vectors = linalg.eigh(S1)
        
        # Sort descending
        ix = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[ix]
        eigen_vectors = eigen_vectors[:, ix]
        
        # Spatial filters W = B^T * P
        self.filters_ = np.dot(eigen_vectors.T, whitening)
        self.eigen_values_ = eigen_values
        
        # Spatial patterns (inverse of filters)
        self.patterns_ = linalg.pinv(self.filters_)
        
        print(f"CSP fitted: {self.n_components} components selected")
        return self
    
    def transform(self, X):
        if self.filters_ is None:
            raise ValueError("CSP must be fitted first. Call fit() before transform()")
        
        n_trials, n_channels, n_samples = X.shape
        
        # Select m filters from each end
        m = self.n_components // 2
        selected_filters = np.vstack([
            self.filters_[:m],      # First m (class 1)
            self.filters_[-m:]      # Last m (class 2)
        ])
        
        features = np.zeros((n_trials, self.n_components))
        
        for trial in range(n_trials):
            # Project: Z = W * X
            projected = np.dot(selected_filters, X[trial])
            
            # Compute variance per component
            variance = np.var(projected, axis=1)
            
            # Normalize
            variance = variance / np.sum(variance)
            
            # Log transform if specified
            if self.log:
                # Add small epsilon to avoid log(0)
                features[trial] = np.log(variance + 1e-10)
            else:
                features[trial] = variance
        
        print(f"CSP features shape: {features.shape}")
        return features
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)