"""Unit tests for src/preprocess.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from src.preprocess import clean_text, preprocess_dataframe, split_data


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO WORLD", stemming=False) == "hello world"

    def test_removes_url(self):
        result = clean_text("Visit http://spam.com now", stemming=False, remove_stopwords=False)
        assert "http" not in result

    def test_removes_email(self):
        result = clean_text("Contact us at free@prize.com", stemming=False)
        assert "@" not in result

    def test_removes_subject_header(self):
        result = clean_text("Subject: Win a prize!", stemming=False)
        assert "Subject" not in result

    def test_removes_stopwords(self):
        result = clean_text("the cat sat on the mat", stemming=False)
        assert "the" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string(self):
        assert clean_text(None) == ""

    def test_min_word_length(self):
        result = clean_text("a ab abc abcd", stemming=False, remove_stopwords=False, min_word_length=3)
        tokens = result.split()
        assert all(len(t) >= 3 for t in tokens)


class TestPreprocessDataframe:
    def test_adds_clean_text_column(self):
        df = pd.DataFrame({"text": ["Hello World", "Free prize now!"], "spam": [0, 1]})
        out = preprocess_dataframe(df)
        assert "clean_text" in out.columns

    def test_original_preserved(self):
        df = pd.DataFrame({"text": ["Hello World"], "spam": [0]})
        out = preprocess_dataframe(df)
        assert "text" in out.columns

    def test_no_nulls_in_output(self):
        df = pd.DataFrame({"text": ["Hello", None, "Spam"], "spam": [0, 0, 1]})
        out = preprocess_dataframe(df)
        assert out["clean_text"].notna().all()


class TestSplitData:
    def test_split_ratio(self):
        df = pd.DataFrame({
            "clean_text": ["email"] * 100,
            "spam": [0] * 76 + [1] * 24,
        })
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        assert len(X_test) == pytest.approx(20, abs=2)
        assert len(X_train) == pytest.approx(80, abs=2)

    def test_stratified(self):
        df = pd.DataFrame({
            "clean_text": ["email"] * 100,
            "spam": [0] * 76 + [1] * 24,
        })
        _, _, y_train, y_test = split_data(df, test_size=0.2)
        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05