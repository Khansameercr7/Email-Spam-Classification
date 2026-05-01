"""
predict.py
----------
Load the saved model + vectoriser and classify raw email text.

Usage (CLI):
    python src/predict.py "Subject: Congratulations! You have won a prize!"
    python src/predict.py --file path/to/email.txt

Usage (Python API):
    from src.predict import SpamPredictor
    p = SpamPredictor()
    print(p.predict("Free money, click now!"))   # → "SPAM"
    print(p.predict_batch(["Hello", "Win $$"])) # → ["HAM", "SPAM"]
"""

import os
import sys
import yaml
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import clean_text


class SpamPredictor:
    """Wraps the saved vectoriser + classifier for easy inference."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.vectorizer = joblib.load(cfg["paths"]["vectorizer_path"])
        self.model      = joblib.load(cfg["paths"]["model_path"])
        self.pp         = cfg["preprocessing"]

    def _clean(self, text: str) -> str:
        return clean_text(
            text,
            lowercase=self.pp["lowercase"],
            remove_punctuation=self.pp["remove_punctuation"],
            remove_stopwords=self.pp["remove_stopwords"],
            stemming=self.pp["stemming"],
            min_word_length=self.pp["min_word_length"],
        )

    def predict(self, text: str) -> str:
        """Return 'SPAM' or 'HAM' for a single email string."""
        cleaned = self._clean(text)
        vec     = self.vectorizer.transform([cleaned])
        label   = self.model.predict(vec)[0]
        return "SPAM" if label == 1 else "HAM"

    def predict_proba(self, text: str) -> dict:
        """
        Return probability scores for HAM and SPAM.
        Uses calibration for LinearSVC or predict_proba for other models.
        """
        cleaned = self._clean(text)
        vec     = self.vectorizer.transform([cleaned])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            return {"HAM": round(probs[0], 4), "SPAM": round(probs[1], 4)}
        elif hasattr(self.model, "decision_function"):
            # Convert decision function to probability using sigmoid
            score = self.model.decision_function(vec)[0]
            # Sigmoid function to convert to probability
            spam_prob = 1 / (1 + np.exp(-score))
            ham_prob = 1 - spam_prob
            return {"HAM": round(ham_prob, 4), "SPAM": round(spam_prob, 4)}
        return {"HAM": 0.5, "SPAM": 0.5}

    def predict_batch(self, texts: list) -> list:
        """Classify a list of raw email strings."""
        cleaned = [self._clean(t) for t in texts]
        vecs    = self.vectorizer.transform(cleaned)
        labels  = self.model.predict(vecs)
        return ["SPAM" if l == 1 else "HAM" for l in labels]


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Email Spam Classifier — inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("text", nargs="?", help="Raw email text (quoted string)")
    group.add_argument("--file", help="Path to a plain-text email file")
    args = parser.parse_args()

    predictor = SpamPredictor()

    if args.file:
        with open(args.file) as f:
            raw = f.read()
    else:
        raw = args.text

    result = predictor.predict(raw)
    proba  = predictor.predict_proba(raw)
    print(f"\nPrediction : {result}")
    print(f"Confidence : {proba}")