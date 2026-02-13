
import joblib
import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import re
import time
from typing import Any, Optional
import nltk
import numpy as np
from ml.pipeline_component import VariationResponseEncoder

# Ensure stopwords are available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str, stop_words: set[str]) -> str:
    if not isinstance(text, str):
        return text

    text = re.sub(r"[^a-zA-Z0-9\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    tokens = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)

def create_processed_data(df):
    """
    Create a processed dataframe for EDA.
    - Replaces 'Text' with cleaned text
    - Raw 'Text' remains preserved in the original dataframe
    """

    start_time = time.perf_counter()

    processed_df = df.copy()

    # Replace Text with cleaned version
    processed_df["Text"] = processed_df["Text"].apply(
        lambda x: clean_text(x, STOP_WORDS)
    )

    elapsed = time.perf_counter() - start_time
    print(f"Text preprocessing completed in {elapsed:.2f} seconds")
    return processed_df

def load_pipeline(path: str) -> Pipeline:
    pipeline = joblib.load(path)
    return pipeline

def single_inference(pipeline: Any, query: Any, out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Run pipeline.predict / pipeline.predict_proba on input query and return DataFrame with ID, pred_class, prob_class_1..K.

    query may be:
      - pandas.DataFrame containing columns: Gene, Variation, Text (ID optional)
      - dict of lists: {"Gene":[...], "Variation":[...], "Text":[...], "ID":[...](optional)}
      - list of dicts: [{"Gene":"g1","Variation":"v1","Text":"t1", "ID":...}, {...}, ...]

    Returns:
      pd.DataFrame with columns: ID, pred_class, prob_class_1 .. prob_class_K
    """
    # Normalize input to DataFrame
    if isinstance(query, pd.DataFrame):
        df = query.copy()
    elif isinstance(query, dict):
        # dict-of-lists expected
        df = pd.DataFrame(query)
    elif isinstance(query, (list, tuple)):
        # list of dicts expected
        df = pd.DataFrame(list(query))
    else:
        raise TypeError("query must be a pandas.DataFrame, dict-of-lists, or list-of-dicts")

    # Validate required feature columns
    required_feats = {"Gene", "Variation", "Text"}
    if not required_feats.issubset(df.columns):
        missing = required_feats - set(df.columns)
        raise ValueError(f"Input is missing required columns: {missing}")

    # Ensure columns are aligned and lengths consistent
    n = len(df)
    if n == 0:
        # Return empty DataFrame with the intended columns if we can determine number of classes from pipeline
        try:
            n_classes = pipeline.predict_proba(pd.DataFrame([{ "Gene": "", "Variation": "", "Text": "" }])) .shape[1]
        except Exception:
            n_classes = None
        prob_cols = [f"prob_class_{i+1}" for i in range(n_classes)] if n_classes else []
        out_df = pd.DataFrame(columns=["ID", "pred_class"] + prob_cols)
        if out_csv:
            out_df.to_csv(out_csv, index=False)
        return out_df

    # Create ID column if missing
    if "ID" not in df.columns:
        df.insert(0, "ID", [f"auto_{i}" for i in range(n)])

    # Select feature matrix
    X = df[["Gene", "Variation", "Text"]]

    # Run predictions
    preds = pipeline.predict(X)                    # shape (n,)
    probs = pipeline.predict_proba(X)              # shape (n, n_classes)

    # Build output DataFrame of probabilities
    prob_cols = [f"prob_class_{i+1}" for i in range(probs.shape[1])]
    probs_df = pd.DataFrame(probs, columns=prob_cols, index=df.index)

    out_df = pd.DataFrame({
        "ID": df["ID"].values,
        "pred_class": preds
    }, index=df.index)

    # Concatenate probability columns to the right of pred_class
    out_df = pd.concat([out_df, probs_df], axis=1)

    if out_csv:
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote predictions to: {out_csv}")

    return out_df

def _is_empty_cell(val) -> bool:
    """True if val is None, NaN, or a string that's empty/whitespace after strip()."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip() == ""


# -------------------------
# Prediction function (core)
# -------------------------
def batch_inference(pipeline: Any, query: Any, out_csv: Optional[str] = None) -> Any:
    """
    Runs predictions on the provided `query` and returns a pandas DataFrame with columns:
      ID, pred_class, prob_class_1 .. prob_class_K

    If the input is empty or is a single-row with all three feature cells empty/whitespace/NaN,
    this function will (by design) write an empty out_csv if requested and return "".

    Accepts:
      - query: pandas.DataFrame (preferred) containing columns: ID (optional), Gene, Variation, Text
               OR a dict-of-lists OR a list-of-dicts (converted to DataFrame inside)
    """
    # Normalize input to DataFrame
    if isinstance(query, pd.DataFrame):
        df = query.copy()
    elif isinstance(query, dict):
        df = pd.DataFrame(query)
    elif isinstance(query, (list, tuple)):
        df = pd.DataFrame(list(query))
    else:
        raise TypeError("query must be a pandas.DataFrame, dict-of-lists, or list-of-dicts")

    # Validate presence of required features
    required_feats = {"Gene", "Variation", "Text"}
    if not required_feats.issubset(df.columns):
        missing = required_feats - set(df.columns)
        raise ValueError(f"Input is missing required columns: {missing}")

    # Determine empty or single-empty-row condition
    n = len(df)
    single_all_empty = False
    if n == 1:
        row_idx = df.index[0]
        if (_is_empty_cell(df.at[row_idx, "Gene"])
                and _is_empty_cell(df.at[row_idx, "Variation"])
                and _is_empty_cell(df.at[row_idx, "Text"])):
            single_all_empty = True

    if n == 0 or single_all_empty:
        # create empty output file if requested (so downstream callers expecting a file won't error)
        if out_csv:
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("")  # empty file
        # return empty string per your earlier requirement
        return ""

    # Ensure ID exists
    if "ID" not in df.columns:
        df.insert(0, "ID", [f"auto_{i}" for i in range(n)])

    # Select features
    X = df[["Gene", "Variation", "Text"]]

    # Run predictions
    preds = pipeline.predict(X)                    # shape (n,)
    probs = pipeline.predict_proba(X)              # shape (n, n_classes)

    # Build output DataFrame
    prob_cols = [f"prob_class_{i+1}" for i in range(probs.shape[1])]
    probs_df = pd.DataFrame(probs, columns=prob_cols, index=df.index)

    out_df = pd.DataFrame({
        "ID": df["ID"].values,
        "pred_class": preds
    }, index=df.index)

    out_df = pd.concat([out_df, probs_df], axis=1)

    if out_csv:
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote predictions to: {out_csv}")

    return out_df
