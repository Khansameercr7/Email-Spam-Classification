"""
preprocess.py
-------------
Cleans and normalises raw email text before feature extraction.
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Use sklearn's built-in English stopwords (no network download needed)
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Lightweight Porter stemmer (pure Python, no NLTK needed)
class _SimpleStemmer:
    """Minimal suffix-stripping stemmer (good enough for spam classification)."""
    _suffixes = ["ing", "tion", "tions", "ness", "ment", "ments",
                 "ful", "less", "ly", "ed", "er", "est", "ers"]

    def stem(self, word: str) -> str:
        for sfx in self._suffixes:
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        return word

STEMMER = _SimpleStemmer()


def clean_text(text: str,
               lowercase: bool = True,
               remove_punctuation: bool = True,
               remove_stopwords: bool = True,
               stemming: bool = True,
               min_word_length: int = 2) -> str:
    """
    Full text cleaning pipeline for a single email string.

    Parameters
    ----------
    text               : raw email string
    lowercase          : convert to lowercase
    remove_punctuation : strip punctuation and digits
    remove_stopwords   : drop NLTK English stopwords
    stemming           : apply Porter stemmer
    min_word_length    : drop tokens shorter than this

    Returns
    -------
    Cleaned string ready for vectorisation.
    """
    if not isinstance(text, str):
        return ""

    # Remove email headers (Subject: …)
    text = re.sub(r"Subject:\s*", "", text, flags=re.IGNORECASE)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation and digits
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Tokenise on whitespace
    tokens = text.split()

    # Remove stopwords and short tokens
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= min_word_length]
    else:
        tokens = [t for t in tokens if len(t) >= min_word_length]

    # Stemming
    if stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "text",
                         **kwargs) -> pd.DataFrame:
    """
    Apply clean_text to every row of a DataFrame.

    Parameters
    ----------
    df       : DataFrame with a text column
    text_col : name of the column to clean
    **kwargs : forwarded to clean_text

    Returns
    -------
    DataFrame with an added 'clean_text' column and additional features.
    """
    df = df.copy()
    
    # Convert to string and handle None/NaN values
    text_series = df[text_col].fillna("").astype(str)
    
    # Add email length features before cleaning
    df["email_length"] = text_series.apply(len)
    df["word_count"] = text_series.apply(lambda x: len(x.split()))
    df["has_numbers"] = text_series.str.contains(r'\d').astype(int)
    df["has_url"] = text_series.str.contains(r'http|www', case=False).astype(int)
    df["has_exclamation"] = text_series.str.contains('!').astype(int)
    df["has_caps_words"] = text_series.apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    
    # Apply text cleaning
    df["clean_text"] = df[text_col].apply(lambda x: clean_text(x, **kwargs))
    return df


def split_data(df: pd.DataFrame,
               text_col: str = "clean_text",
               label_col: str = "spam",
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all as Series / arrays)
    """
    from sklearn.model_selection import train_test_split

    X = df[text_col]
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


if __name__ == "__main__":
    import yaml, os

    cfg = yaml.safe_load(open("config.yaml"))
    raw = pd.read_csv(cfg["data"]["raw_path"])
    print(f"Loaded {len(raw):,} rows")

    cleaned = preprocess_dataframe(
        raw,
        text_col=cfg["data"]["text_column"],
        lowercase=cfg["preprocessing"]["lowercase"],
        remove_punctuation=cfg["preprocessing"]["remove_punctuation"],
        remove_stopwords=cfg["preprocessing"]["remove_stopwords"],
        stemming=cfg["preprocessing"]["stemming"],
        min_word_length=cfg["preprocessing"]["min_word_length"],
    )

    X_train, X_test, y_train, y_test = split_data(
        cleaned,
        label_col=cfg["data"]["label_column"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    os.makedirs("data", exist_ok=True)
    train_df = pd.DataFrame({"clean_text": X_train, "spam": y_train})
    test_df  = pd.DataFrame({"clean_text": X_test,  "spam": y_test})
    train_df.to_csv(cfg["data"]["train_path"], index=False)
    test_df.to_csv(cfg["data"]["test_path"],   index=False)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print("Saved to data/train.csv and data/test.csv")