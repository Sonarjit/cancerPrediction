import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import re
import time
from typing import Any, Optional, Dict

# Ensure stopwords are available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


# -------------------------
# Text cleaning utilities
# -------------------------
def clean_text(text: str, stop_words: set[str]) -> str:
    if not isinstance(text, str):
        return text

    # Keep only letters, numbers and newlines; collapse whitespace; lowercase
    text = re.sub(r"[^a-zA-Z0-9\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def create_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where the 'Text' column is replaced with cleaned text.
    Assumes df contains 'Text' column.
    """
    start_time = time.perf_counter()
    processed_df = df.copy()
    processed_df["Text"] = processed_df["Text"].apply(lambda x: clean_text(x, STOP_WORDS))
    elapsed = time.perf_counter() - start_time
    print(f"Text preprocessing completed in {elapsed:.2f} seconds")
    return processed_df


# -------------------------
# Pipeline loading
# -------------------------
def load_pipeline(path: str) -> Pipeline:
    """
    Load a joblib pipeline. Caller must ensure any custom transformer classes are importable.
    """
    pipeline = joblib.load(path)
    return pipeline


# -------------------------
# Helper: empty-cell check
# -------------------------
import numpy as np


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
def predict_with_pipeline(pipeline: Any, query: Any, out_csv: Optional[str] = None) -> Any:
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


# -------------------------
# Top-level CSV-based entry
# -------------------------
def predict_from_csv(
    input_csv_path: str,
    pipeline_path: str = "ml/pipeline.joblib",
    out_csv: Optional[str] = "../predictions.csv",
) -> Any:
    # 1) Read CSV
    df = pd.read_csv(input_csv_path)

    # Validate columns
    required_cols = {"ID", "Gene", "Variation", "Text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    # 2) Preprocess text (applies cleaning to Text column)
    processed_df = create_processed_data(df[["ID", "Gene", "Variation", "Text"]])

    # 3) Load pipeline
    pipeline = load_pipeline(pipeline_path)

    # 4) Predict (returns DataFrame or empty string)
    df_out = predict_with_pipeline(pipeline, query=processed_df, out_csv=out_csv)

    # DEBUG: always safe to print head only if df_out is a DataFrame
    if isinstance(df_out, pd.DataFrame):
        # print("Prediction DataFrame:\n", df_out.head())
        pass
    else:
        print("predict_with_pipeline returned non-DataFrame:", repr(df_out))

    # Robustly detect empty-case sentinel
    if isinstance(df_out, str):
        if df_out == "":
            print("Returning empty string per your request")
            return ""
        raise ValueError(f"predict_with_pipeline returned unexpected string: {df_out!r}")

    # At this point df_out should be a DataFrame
    if not isinstance(df_out, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or empty-string from predict_with_pipeline, got {type(df_out)}")

    # 5) Convert DataFrame to requested dict shape
    print("Converting prediction DataFrame to output dictionary format")
    result: Dict[str, list] = {}
    result["ID"] = df_out["ID"].tolist()
    result["predicted_class"] = df_out["pred_class"].astype(int).tolist()

    prob_cols = [c for c in df_out.columns if c.startswith("prob_class_")]
    prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split("_")[-1]))
    for idx, col in enumerate(prob_cols_sorted, start=1):
        key = f"class{idx}_prob"
        result[key] = df_out[col].astype(float).tolist()

    return {"result": result}


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # # Example: user supplies "user_input.csv" in same folder with required columns
    # output = predict_from_csv("variants.csv", pipeline_path="ml/pipeline.joblib", out_csv="../predictions.csv")
    # print(output)
    pass
