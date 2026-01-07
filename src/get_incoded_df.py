
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_split import get_train_data, get_cv_data, get_test_data

# LOAD DATA
train_df = get_train_data()
test_df = get_test_data()
cv_df = get_cv_data()
d_train_gene = train_df['Gene']

#================ GENE FEATURE ENCODING ===================
# ONE HOT ENCODING
gene_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

train_gene_ohe = gene_encoder.fit_transform(train_df[["Gene"]])
test_gene_ohe = gene_encoder.transform(test_df[["Gene"]])
cv_gene_ohe = gene_encoder.transform(cv_df[["Gene"]])

# RESPONSE CODING
gene_class_dict = defaultdict(list)

for gene, cls in zip(train_df["Gene"], train_df["Class"]):
    gene_class_dict[gene].append(cls)

NUM_CLASSES = 9
alpha = 1  # Laplace smoothing

gene_response_dict = {}

for gene, class_list in gene_class_dict.items():
    counts = np.zeros(NUM_CLASSES)

    for cls in class_list:
        counts[cls - 1] += 1   # class labels 1–9 → index 0–8

    # Laplace smoothing + normalization
    probs = (counts + alpha*10) / (counts.sum() + alpha * 90)

    gene_response_dict[gene] = probs

global_class_probs = (
    train_df["Class"]
    .value_counts(normalize=True)
    .sort_index()
    .values
)

def response_encode_gene(series, response_dict, fallback):
    return np.vstack([
        response_dict.get(gene, fallback)
        for gene in series
    ])

X_train_gene_rc = response_encode_gene(
    train_df["Gene"], gene_response_dict, global_class_probs
)

X_test_gene_rc = response_encode_gene(
    test_df["Gene"], gene_response_dict, global_class_probs
)

X_cv_gene_rc = response_encode_gene(
    cv_df["Gene"], gene_response_dict, global_class_probs
)

def get_gene_ohe():
    return train_gene_ohe, test_gene_ohe, cv_gene_ohe
def get_gene_rc():
    return X_train_gene_rc, X_test_gene_rc, X_cv_gene_rc
#======================================================
#================ VARIATION FEATURE ENCODING ===================
# RESPONSE CODING
variation_class_dict = defaultdict(list)

for variation, cls in zip(train_df["Variation"], train_df["Class"]):
    variation_class_dict[variation].append(cls)

NUM_CLASSES = 9
alpha = 1  # Laplace smoothing

variation_response_dict = {}

for variation, class_list in variation_class_dict.items():
    counts = np.zeros(NUM_CLASSES)

    for cls in class_list:
        counts[cls - 1] += 1   # class labels 1–9 → index 0–8

    # Laplace smoothing + normalization
    probs = (counts + alpha*10) / (counts.sum() + alpha * 90)

    variation_response_dict[variation] = probs

def response_encode_variation(series, response_dict, fallback):
    return np.vstack([
        response_dict.get(variation, fallback)
        for variation in series
    ])

X_train_variation_rc = response_encode_variation(
    train_df["Variation"], variation_response_dict, global_class_probs
)

X_test_variation_rc = response_encode_variation(
    test_df["Variation"], variation_response_dict, global_class_probs
)

X_cv_variation_rc = response_encode_variation(
    cv_df["Variation"], variation_response_dict, global_class_probs
)
def get_variation_rc():
    return X_train_variation_rc, X_test_variation_rc, X_cv_variation_rc
#======================================================
#================ TEXT FEATURE ENCODING ===================
bow_vectorizer = CountVectorizer(
    min_df=3,          # ignore very rare words
    max_df=0.9,        # ignore extremely frequent words
    ngram_range=(1, 1) # unigrams
)

# Fit ONLY on training data
X_train_bow = bow_vectorizer.fit_transform(train_df["Text"])

# Transform test and CV
X_test_bow = bow_vectorizer.transform(test_df["Text"])
X_cv_bow = bow_vectorizer.transform(cv_df["Text"])



scaler = StandardScaler(with_mean=False)  # sparse-safe

X_train_bow_scaled = scaler.fit_transform(X_train_bow)
X_test_bow_scaled = scaler.transform(X_test_bow)
X_cv_bow_scaled = scaler.transform(X_cv_bow)

tfidf_vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words="english" # optional
)

# Fit ONLY on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["Text"])

# Transform test and CV
X_test_tfidf = tfidf_vectorizer.transform(test_df["Text"])
X_cv_tfidf = tfidf_vectorizer.transform(cv_df["Text"])

def get_text_bow():
    return X_train_bow_scaled, X_test_bow_scaled, X_cv_bow_scaled
def get_text_tfidf():
    return X_train_tfidf, X_test_tfidf, X_cv_tfidf
#======================================================