"""
Integration tests for src/predict.py
Runs the full pipeline (train → save → predict) using a tiny synthetic dataset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import tempfile
import shutil
import pandas as pd
import yaml

from src.preprocess import preprocess_dataframe
from src.features   import build_vectorizer, fit_transform, transform, save_vectorizer
from sklearn.linear_model import LogisticRegression
import joblib


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Train a tiny model and return paths to model + vectorizer."""
    tmp = tmp_path_factory.mktemp("artifacts")

    # Tiny dataset
    spam_texts = [
        "win free money prize lottery now",
        "congratulations you have won a prize",
        "free offer click here cash reward",
        "urgent claim your prize today",
        "make money fast free bonus",
    ] * 4

    ham_texts = [
        "hello how are you doing today",
        "meeting tomorrow at 10 am",
        "please review the attached report",
        "thanks for your help with the project",
        "see you at the conference next week",
    ] * 4

    df = pd.DataFrame({
        "text": spam_texts + ham_texts,
        "spam": [1] * len(spam_texts) + [0] * len(ham_texts),
    })

    cleaned = preprocess_dataframe(df)
    vec = build_vectorizer(method="tfidf", max_features=500, ngram_range=(1, 1), sublinear_tf=True)
    X   = fit_transform(vec, cleaned["clean_text"])
    y   = cleaned["spam"]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    vec_path   = str(tmp / "vectorizer.pkl")
    model_path = str(tmp / "spam_classifier.pkl")
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)

    # Write a config pointing to these tmp paths
    cfg = {
        "paths": {"vectorizer_path": vec_path, "model_path": model_path},
        "preprocessing": {
            "lowercase": True, "remove_punctuation": True,
            "remove_stopwords": True, "stemming": True, "min_word_length": 2,
        },
    }
    cfg_path = str(tmp / "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    return cfg_path


class TestSpamPredictor:
    def test_predict_returns_label(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        result = p.predict("hello how are you")
        assert result in ("SPAM", "HAM")

    def test_obvious_spam(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        result = p.predict("win free money prize lottery now congratulations")
        assert result == "SPAM"

    def test_obvious_ham(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        result = p.predict("hello meeting tomorrow please review the report")
        assert result == "HAM"

    def test_predict_batch(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        results = p.predict_batch(["free money prize", "hello how are you"])
        assert len(results) == 2
        assert all(r in ("SPAM", "HAM") for r in results)

    def test_predict_proba_keys(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        proba = p.predict_proba("free money now")
        assert "SPAM" in proba or "decision_score" in proba

    def test_empty_string(self, trained_artifacts):
        from src.predict import SpamPredictor
        p = SpamPredictor(config_path=trained_artifacts)
        result = p.predict("")
        assert result in ("SPAM", "HAM")