# pipeline_build.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from collections import defaultdict

NUM_CLASSES = 9

class VariationResponseEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that maps Variation -> response probability vector (length NUM_CLASSES).
    Stores mapping fitted on training data and a fallback global distribution.
    Returns a 2D numpy array (n_samples, NUM_CLASSES).
    """
    def __init__(self, alpha=1, num_classes=NUM_CLASSES):
        self.alpha = alpha
        self.num_classes = num_classes
        self.variation_to_probs = {}
        self.fallback = None

    def fit(self, X, y=None):
        # X: iterable of variations
        # y: matching class labels (1..NUM_CLASSES)
        var_counts = defaultdict(lambda: np.zeros(self.num_classes, dtype=float))
        global_counts = np.zeros(self.num_classes, dtype=float)
        for variation, cls in zip(X, y):
            if pd.isna(variation):
                variation = "__MISSING__"
            idx = int(cls) - 1
            var_counts[variation][idx] += 1
            global_counts[idx] += 1

        # create smoothed probabilities
        alpha = self.alpha * 10  # you used alpha*10 in train code
        denom_add = self.alpha * (self.num_classes * 10)  # matches your earlier code
        for var, counts in var_counts.items():
            probs = (counts + alpha) / (counts.sum() + denom_add)
            self.variation_to_probs[var] = probs

        # global fallback
        self.fallback = (global_counts + alpha) / (global_counts.sum() + denom_add)
        return self

    def transform(self, X):
        out = []
        for v in X:
            if pd.isna(v):
                v = "__MISSING__"
            out.append(self.variation_to_probs.get(v, self.fallback))
        return np.vstack(out)  # shape (n_samples, NUM_CLASSES)

# Example pipeline assembly
def build_and_train_pipeline(train_df):
    # train_df must have columns: 'Gene','Variation','Text','Class'
    gene_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    text_tfidf = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1,2), stop_words="english"
    )
    variation_rc = VariationResponseEncoder(alpha=1, num_classes=NUM_CLASSES)

    # ColumnTransformer expects a 2D input or DataFrame; pass DataFrame columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("gene", gene_ohe, ["Gene"]),
            # VariationResponseEncoder will receive a 1-column array-like
            ("variation", variation_rc, ["Variation"]),
            # TfidfVectorizer expects raw text; ColumnTransformer will pass the column
            ("text", text_tfidf, "Text"),
        ],
        remainder="drop",
        sparse_threshold=0.0  # keep sparse where possible
    )

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", clf)
    ])

    pipeline.fit(train_df[["Gene","Variation","Text"]], train_df["Class"])
    return pipeline

# Save pipeline
# pipeline = build_and_train_pipeline(train_df)
# joblib.dump(pipeline, "cancer_pipeline.joblib")
