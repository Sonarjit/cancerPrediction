# pipeline_components.py
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

NUM_CLASSES = 9

class VariationResponseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_classes: int = NUM_CLASSES, alpha: float = 1.0):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.variation_to_probs = {}
        self.fallback_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y required for fit")
        variations = self._to_1d_list(X)
        labels = np.asarray(y)
        var_counts = defaultdict(lambda: np.zeros(self.num_classes, dtype=float))
        global_counts = np.zeros(self.num_classes, dtype=float)
        for var, lbl in zip(variations, labels):
            key = "__MISSING__" if (var is None or (isinstance(var, float) and np.isnan(var))) else str(var)
            idx = int(lbl) - 1
            var_counts[key][idx] += 1
            global_counts[idx] += 1
        alpha_term = self.alpha * 10.0
        denom_add = self.alpha * (self.num_classes * 10.0)
        for k, counts in var_counts.items():
            self.variation_to_probs[k] = (counts + alpha_term) / (counts.sum() + denom_add)
        self.fallback_ = (global_counts + alpha_term) / (global_counts.sum() + denom_add)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Not fitted.")
        variations = self._to_1d_list(X)
        rows = []
        for var in variations:
            key = "__MISSING__" if (var is None or (isinstance(var, float) and np.isnan(var))) else str(var)
            rows.append(self.variation_to_probs.get(key, self.fallback_))
        arr = np.vstack(rows)
        return sparse.csr_matrix(arr)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    @staticmethod
    def _to_1d_list(X):
        if X is None:
            return []
        if isinstance(X, pd.Series):
            return X.tolist()
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].tolist()
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:,0].tolist()
        return arr.ravel().tolist()
