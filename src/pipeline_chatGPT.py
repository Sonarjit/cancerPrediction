"""
pipeline.py

Usage:
    - Train and save:
        python pipeline.py train --input_csv path/to/train.csv --out pipeline.joblib

    - Predict using saved pipeline:
        python pipeline.py predict --pipeline pipeline.joblib --input_csv path/to/new_rows.csv --out preds.csv

Assumptions:
    - Input CSV has columns: ID, Gene, Variation, Text, Class (Class only required for training)
    - Stop-word removal / other text cleaning has already been applied before splitting (per user's instruction).
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Configuration / constants
# ----------------------------
NUM_CLASSES = 9
VARIATION_ALPHA = 1  # Laplace smoothing base (kept consistent with your formula)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 2)


# ----------------------------
# Custom transformer
# ----------------------------
class VariationResponseEncoder(BaseEstimator, TransformerMixin):
    """
    Fit-time:
        - Build mapping: variation -> smoothed probability vector (length NUM_CLASSES)
        - Build global fallback probability vector (train distribution with same smoothing)

    Transform-time:
        - For each variation value, emit the corresponding probability vector (shape n_samples x NUM_CLASSES)
        - Unknown variation -> global fallback

    Important:
        - Accepts X in any common array-like shape (pandas Series, 1-column DataFrame, nested arrays).
        - Returns a scipy.sparse.csr_matrix of shape (n_samples, num_classes).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, alpha: float = VARIATION_ALPHA):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.variation_to_probs: Dict[str, np.ndarray] = {}
        self.fallback_: Optional[np.ndarray] = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        X: array-like of variations (n_samples, ) or (n_samples, 1)
        y: array-like of class labels (1..NUM_CLASSES) (required)
        """
        if y is None:
            raise ValueError("VariationResponseEncoder requires y (class labels) on fit.")

        # Normalize X to 1D list of strings
        variations = self._to_1d_list(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.shape[0] != len(variations):
            raise ValueError("y must be 1D array-like and match X length")

        # Build counts per variation
        var_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.num_classes, dtype=float))
        global_counts = np.zeros(self.num_classes, dtype=float)

        for var, lbl in zip(variations, labels):
            # convert missing to sentinel
            if var is None or (isinstance(var, float) and np.isnan(var)):
                var_key = "__MISSING__"
            else:
                var_key = str(var)
            idx = int(lbl) - 1  # labels are 1..NUM_CLASSES -> indices 0..NUM_CLASSES-1
            if not (0 <= idx < self.num_classes):
                raise ValueError(f"Label {lbl} outside expected 1..{self.num_classes}")
            var_counts[var_key][idx] += 1
            global_counts[idx] += 1

        # Smoothing behaviour to match your original code:
        # per-variation: (counts + alpha*10) / (counts.sum() + alpha * 90)
        # This uses alpha*10 and alpha*90 as in your code snippet.
        alpha_term = self.alpha * 10.0
        denom_add = self.alpha * (self.num_classes * 10.0)  # 90 when num_classes=9

        for var_key, counts in var_counts.items():
            probs = (counts + alpha_term) / (counts.sum() + denom_add)
            # Ensure numeric stability
            probs = np.asarray(probs, dtype=float)
            self.variation_to_probs[var_key] = probs

        # Global fallback
        self.fallback_ = (global_counts + alpha_term) / (global_counts.sum() + denom_add)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("VariationResponseEncoder is not fitted. Call fit(X, y) first.")

        variations = self._to_1d_list(X)
        rows = []
        for var in variations:
            if var is None or (isinstance(var, float) and np.isnan(var)):
                key = "__MISSING__"
            else:
                key = str(var)
            probs = self.variation_to_probs.get(key, self.fallback_)
            rows.append(probs)

        arr = np.vstack(rows)  # shape (n_samples, num_classes)
        # Convert to sparse CSR to keep memory efficiency when combining with sparse TF-IDF / OHE
        return sparse.csr_matrix(arr)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def _to_1d_list(X):
        # Accept pandas Series, 1-column DataFrame, numpy arrays, lists
        if X is None:
            return []
        if isinstance(X, pd.Series):
            return X.tolist()
        if isinstance(X, pd.DataFrame):
            # take first column
            if X.shape[1] >= 1:
                return X.iloc[:, 0].tolist()
            return []
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0].tolist()
        return arr.ravel().tolist()


# ----------------------------
# Pipeline construction
# ----------------------------
def build_pipeline(
    num_classes: int = NUM_CLASSES,
    variation_alpha: float = VARIATION_ALPHA,
    tfidf_min_df: int = TFIDF_MIN_DF,
    tfidf_max_df: float = TFIDF_MAX_DF,
    tfidf_ngram_range: tuple = TFIDF_NGRAM_RANGE,
) -> Pipeline:
    """
    Build and return a scikit-learn Pipeline:
        ColumnTransformer(
            gene -> OneHotEncoder,
            variation -> VariationResponseEncoder,
            text -> TfidfVectorizer
        ) -> CalibratedClassifierCV(SGDClassifier)
    """

    gene_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    variation_encoder = VariationResponseEncoder(num_classes=num_classes, alpha=variation_alpha)
    tfidf = TfidfVectorizer(min_df=tfidf_min_df, max_df=tfidf_max_df, ngram_range=tfidf_ngram_range, stop_words=None)

    preprocessor = ColumnTransformer(
        transformers=[
            ("gene_ohe", gene_encoder, ["Gene"]),
            ("variation_rc", variation_encoder, ["Variation"]),
            ("text_tfidf", tfidf, "Text"),  # pass column name directly for DataFrame
        ],
        remainder="drop",  # drop other columns
        sparse_threshold=0.0,  # keep sparse where possible
    )

    base_clf = SGDClassifier(class_weight="balanced", alpha=0.0001, penalty="l2", loss="log_loss", random_state=42, max_iter=1000)
    calibrated = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=5)

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", calibrated),
    ])

    return pipeline


# ----------------------------
# Train / save / load / predict helpers
# ----------------------------
def train_and_save_pipeline(input_csv: str, pipeline_out: str, test_size: float = 0.1, random_state: int = 42):
    """
    Read CSV, train pipeline on training portion, save pipeline artifact and metadata.
    CSV must contain columns: ID, Gene, Variation, Text, Class
    """
    df = pd.read_csv(input_csv)
    required_cols = {"ID", "Gene", "Variation", "Text", "Class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    # Split into train / held-out (optional)
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["Class"], random_state=random_state)

    X_train = train_df[["Gene", "Variation", "Text"]]
    y_train = train_df["Class"]

    pipeline = build_pipeline(num_classes=NUM_CLASSES, variation_alpha=VARIATION_ALPHA)
    print("Fitting pipeline (this may take time for TF-IDF)...")
    pipeline.fit(X_train, y_train)

    # Optionally evaluate on test_df
    X_test = test_df[["Gene", "Variation", "Text"]]
    y_test = test_df["Class"]
    preds = pipeline.predict(X_test)
    print("Evaluation on held-out test (simple):")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save pipeline (joblib)
    joblib.dump(pipeline, pipeline_out)
    # Save metadata
    metadata = {
        "pipeline_path": pipeline_out,
        "num_classes": int(NUM_CLASSES),
        "variation_alpha": float(VARIATION_ALPHA),
        "tfidf_min_df": TFIDF_MIN_DF,
        "tfidf_max_df": TFIDF_MAX_DF,
        "tfidf_ngram_range": TFIDF_NGRAM_RANGE,
    }
    meta_path = pipeline_out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved pipeline to: {pipeline_out}")
    print(f"Saved metadata to: {meta_path}")


def load_pipeline(path: str) -> Pipeline:
    pipeline = joblib.load(path)
    return pipeline


def predict_with_pipeline(pipeline: Pipeline, input_csv: str, out_csv: Optional[str] = None):
    """
    Read input CSV that contains ID, Gene, Variation, Text (Class optional/missing), run pipeline.predict/predict_proba,
    and return a DataFrame with ID, pred_class, prob_1..prob_k
    """
    df = pd.read_csv(input_csv)
    if not {"ID", "Gene", "Variation", "Text"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: ID, Gene, Variation, Text")

    X = df[["Gene", "Variation", "Text"]]
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)  # shape (n_samples, n_classes)

    # Build output DataFrame
    prob_cols = [f"prob_class_{i+1}" for i in range(probs.shape[1])]
    out_df = pd.DataFrame(probs, columns=prob_cols)
    out_df.insert(0, "pred_class", preds)
    out_df.insert(0, "ID", df["ID"].values)

    if out_csv:
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote predictions to: {out_csv}")
    return out_df


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train / use the cancer-class pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train pipeline and save artifact")
    p_train.add_argument("--input_csv", required=True, help="CSV file with ID,Gene,Variation,Text,Class")
    p_train.add_argument("--out", required=True, dest="out_path", help="Output joblib pipeline path")
    p_train.add_argument("--test_size", default=0.1, type=float)
    p_train.add_argument("--random_state", default=42, type=int)

    p_pred = sub.add_parser("predict", help="Load pipeline and predict on new CSV")
    p_pred.add_argument("--pipeline", required=True, help="Path to saved pipeline joblib")
    p_pred.add_argument("--input_csv", required=True, help="CSV with ID,Gene,Variation,Text")
    p_pred.add_argument("--out", required=False, help="Optional CSV path to write predictions")

    args = parser.parse_args()
    if args.cmd == "train":
        train_and_save_pipeline(args.input_csv, args.out_path, test_size=args.test_size, random_state=args.random_state)
    elif args.cmd == "predict":
        pipeline = load_pipeline(args.pipeline)
        df_out = predict_with_pipeline(pipeline, args.input_csv, out_csv=args.out)
        print(df_out.head())


if __name__ == "__main__":
    main()
