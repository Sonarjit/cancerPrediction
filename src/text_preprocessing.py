import re
import time
from nltk.corpus import stopwords

from src.data_loader import get_combine_data


# ---------------------------------------------------------
# Load raw data (immutable)
# ---------------------------------------------------------
combined_data = get_combine_data()


# ---------------------------------------------------------
# Stop words
# ---------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------
# Pure text cleaning function
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Create processed dataframe (NO raw Text column)
# ---------------------------------------------------------
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

def get_processed_data():
    return create_processed_data(combined_data)

# ---------------------------------------------------------
# Script execution (optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    processed_data = create_processed_data(combined_data)
    print(processed_data.head())
