"""
features.py
-----------
Builds and persists the text vectoriser (TF-IDF or Bag-of-Words).
"""

import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_vectorizer(method: str = "tfidf",
                     max_features: int = 8000,
                     ngram_range: tuple = (1, 2),
                     sublinear_tf: bool = True):
    """
    Instantiate a vectoriser based on config.

    Parameters
    ----------
    method       : 'tfidf' or 'bow'
    max_features : vocabulary ceiling
    ngram_range  : unigrams + bigrams by default
    sublinear_tf : apply log(1 + tf) scaling (TF-IDF only)

    Returns
    -------
    Unfitted sklearn vectoriser.
    """
    if method == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
        )
    elif method == "bow":
        return CountVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            strip_accents="unicode",
            analyzer="word",
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tfidf' or 'bow'.")


def fit_transform(vectorizer, X_train):
    """Fit on training data and transform it."""
    return vectorizer.fit_transform(X_train)


def transform(vectorizer, X):
    """Transform new data with an already-fitted vectoriser."""
    return vectorizer.transform(X)


def save_vectorizer(vectorizer, path: str = "models/vectorizer.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: str = "models/vectorizer.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    import yaml, pandas as pd

    cfg = yaml.safe_load(open("config.yaml"))
    fc  = cfg["features"]

    train = pd.read_csv(cfg["data"]["train_path"])
    test  = pd.read_csv(cfg["data"]["test_path"])

    vec = build_vectorizer(
        method=fc["method"],
        max_features=fc["max_features"],
        ngram_range=fc["ngram_range"],
        sublinear_tf=fc["sublinear_tf"],
    )

    X_train = fit_transform(vec, train["clean_text"])
    X_test  = transform(vec, test["clean_text"])

    save_vectorizer(vec, cfg["paths"]["vectorizer_path"])
    print(f"Feature matrix  →  train {X_train.shape}  |  test {X_test.shape}")