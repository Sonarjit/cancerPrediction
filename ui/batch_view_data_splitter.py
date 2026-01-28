import re
import pandas as pd
from typing import Tuple, Dict, Any, List

def _norm_key(k: str) -> str:
    """Lowercase and replace non-alphanumeric with underscore for matching."""
    return re.sub(r'[^0-9a-z]', '_', str(k).lower())

def _find_key(keys: List[str], *patterns) -> str | None:
    """
    Return first key from keys that matches any of the provided normalized patterns.
    Each pattern may be a substring (normalized) or a regex (if starts with 're:').
    """
    norm_keys = {k: _norm_key(k) for k in keys}
    for pat in patterns:
        if isinstance(pat, str) and pat.startswith("re:"):
            regex = re.compile(pat[3:], re.I)
            for original, nk in norm_keys.items():
                if regex.search(nk):
                    return original
        else:
            target = _norm_key(pat)
            for original, nk in norm_keys.items():
                if target == nk or target in nk:
                    return original
    return None

def split_data_for_table(data: Dict[str, List[Any]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[Any]]]]:
    """
    Split a single data dict into (input_df, result_dict) usable by populate_table_from_results.

    Returns:
      input_df: pd.DataFrame with columns ID, Gene, Variation, Text
      result: {"result": { "ID": [...], "predicted_class": [...], "class1_prob":[...], ..., "class9_prob":[...] }}
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict of lists")

    # Ensure all values are lists and have same length
    lengths = set()
    for k, v in data.items():
        if not hasattr(v, "__len__"):
            raise ValueError(f"data[{k!r}] must be a list-like")
        lengths.add(len(v))
    if len(lengths) == 0:
        n = 0
    elif len(lengths) == 1:
        n = lengths.pop()
    else:
        raise ValueError("All lists in data must have the same length")

    keys = list(data.keys())

    # Input columns (flexible matching)
    id_key = _find_key(keys, "id")
    gene_key = _find_key(keys, "gene")
    var_key = _find_key(keys, "variation", "var")
    text_key = _find_key(keys, "text", "description")

    if id_key is None or gene_key is None or var_key is None or text_key is None:
        missing = [name for name, found in (("ID", id_key), ("Gene", gene_key), ("Variation", var_key), ("Text", text_key)) if found is None]
        raise ValueError(f"Input data missing required fields: {missing}. Keys available: {keys}")

    # Build input_df preserving order
    input_df = pd.DataFrame({
        "ID": list(data[id_key]),
        "Gene": list(data[gene_key]),
        "Variation": list(data[var_key]),
        "Text": list(data[text_key])
    })

    # Build result dict
    result_inner: Dict[str, List[Any]] = {}
    result_inner["ID"] = list(data[id_key])

    # Find predicted_class key (many possible names)
    pred_key = _find_key(keys, "predicted_class", "pred_class", "predicted", "prediction")
    if pred_key is None:
        # if not present, create list of empty or zeros
        result_inner["predicted_class"] = [None] * n
    else:
        result_inner["predicted_class"] = list(data[pred_key])

    # For class probabilities 1..9, find keys using flexible matching
    for k in range(1, 10):
        # Try a few candidate patterns:
        # class1_prob, class_1_probability, class1prob, class 1 probability, prob_class_1, probability_class_1
        patterns = [
            f"class{k}_prob", f"class_{k}_prob", f"class_{k}_probability",
            f"class{k}prob", f"class_{k}_probability", f"prob_class_{k}",
            f"probability_class_{k}", f"class_{k}_probability"
        ]
        # also allow regex like r'class.*1.*prob'
        re_pat = f"re:class.*{k}.*(prob|probability|p)"
        found = _find_key(keys, *patterns, re_pat)
        out_key = f"class{k}_prob"
        if found is None:
            # fill zeros if missing
            result_inner[out_key] = [0.0] * n
        else:
            # convert to float (best effort)
            vals = list(data[found])
            # try to coerce numeric values; if cannot, leave as-is
            coerced = []
            for v in vals:
                try:
                    coerced.append(float(v))
                except Exception:
                    coerced.append(v)
            result_inner[out_key] = coerced

    # Return in the wrapper form the method accepts
    return input_df, {"result": result_inner}
