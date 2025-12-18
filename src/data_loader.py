# IMPORT BLOCK
import pandas as pd

# READ TEXT DATA
text_data = pd.read_csv(
    "../training/training_text",
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# READ CLASS DATA
class_data = pd.read_csv("../training/training_variants")
# MERGING THE DATA
combined_data = text_data.merge(
    class_data,
    on="ID",
    how="left"
)

combined_data = combined_data.dropna(subset=["Text"])

# data integrity lock
assert not combined_data["Text"].isna().any()
assert (combined_data["Text"].str.strip() != "").all()


def get_combine_data():
    return combined_data