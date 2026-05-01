"""
train.py
--------
Trains multiple classifiers, prints a comparison table, and saves the best model.
Run from the project root:  python src/train.py
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import cross_val_score, GridSearchCV

# ── local modules ──────────────────────────────────────────────────────────────
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_dataframe, split_data
from src.features   import build_vectorizer, fit_transform, transform, save_vectorizer

MODEL_MAP = {
    "MultinomialNB":         MultinomialNB,
    "LogisticRegression":    LogisticRegression,
    "LinearSVC":             LinearSVC,
    "RandomForestClassifier": RandomForestClassifier,
}


def build_model(model_cfg: dict):
    cls  = MODEL_MAP[model_cfg["type"]]
    return cls(**model_cfg.get("params", {}))


def train_all(cfg: dict):
    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("Loading data …")
    raw = pd.read_csv(cfg["data"]["raw_path"])
    pp  = cfg["preprocessing"]

    cleaned = preprocess_dataframe(
        raw,
        text_col=cfg["data"]["text_column"],
        lowercase=pp["lowercase"],
        remove_punctuation=pp["remove_punctuation"],
        remove_stopwords=pp["remove_stopwords"],
        stemming=pp["stemming"],
        min_word_length=pp["min_word_length"],
    )

    X_train, X_test, y_train, y_test = split_data(
        cleaned,
        label_col=cfg["data"]["label_column"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # Save processed splits
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"clean_text": X_train, "spam": y_train}).to_csv(cfg["data"]["train_path"], index=False)
    pd.DataFrame({"clean_text": X_test,  "spam": y_test }).to_csv(cfg["data"]["test_path"],  index=False)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 2. Vectorise ─────────────────────────────────────────────────────────
    fc  = cfg["features"]
    vec = build_vectorizer(
        method=fc["method"],
        max_features=fc["max_features"],
        ngram_range=fc["ngram_range"],
        sublinear_tf=fc["sublinear_tf"],
    )
    Xtr = fit_transform(vec, X_train)
    Xte = transform(vec, X_test)
    save_vectorizer(vec, cfg["paths"]["vectorizer_path"])

    # ── 3. Train & evaluate all models ───────────────────────────────────────
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>15}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1", "CV F1 (5-fold)"))
    print("─" * 85)

    results = {}
    best_model_name = cfg["training"]["best_model"]

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        clf  = build_model(model_cfg)
        
        # Add class_weight='balanced' for handling class imbalance
        if hasattr(clf, 'class_weight'):
            clf.class_weight = 'balanced'
        elif hasattr(clf, 'set_params'):
            try:
                clf.set_params(class_weight='balanced')
            except:
                pass
        
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(clf, Xtr, y_train, cv=5, scoring="f1")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            cv_str = f"{cv_mean:.4f} ± {cv_scores.std():.4f}"
        except:
            cv_str = "N/A"

        results[name] = {"model": clf, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cv": cv_str}
        marker = " ◀ best" if name == best_model_name else ""
        print("{:<25} {:>9.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>15}{}".format(
            name, acc, prec, rec, f1, cv_str, marker))

    # ── 4. Hyperparameter tuning for best model ───────────────────────────────
    print("\n── Hyperparameter Tuning (GridSearchCV) ──────────────────────────")
    
    best_clf = results[best_model_name]["model"]
    
    # Get the correct model type from config
    model_cfg = None
    for mc in cfg["models"]:
        if mc["name"] == best_model_name:
            model_cfg = mc
            break
    
    clf_type = model_cfg["type"]
    
    # Define parameter grids for each model type
    if clf_type == "LogisticRegression":
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0], "max_iter": [500, 1000]}
    elif clf_type == "MultinomialNB":
        param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0]}
    elif clf_type == "LinearSVC":
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0], "max_iter": [1000, 2000]}
    elif clf_type == "RandomForestClassifier":
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    else:
        param_grid = {}
    
    if param_grid:
        print(f"  Tuning {clf_type} with params: {param_grid}")
        
        # Create fresh model instance for grid search
        base_clf = MODEL_MAP[clf_type](**model_cfg.get("params", {}))
        
        # Add class_weight if supported
        if hasattr(base_clf, 'class_weight'):
            base_clf.class_weight = 'balanced'
        
        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        grid_search.fit(Xtr, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Use best estimator
        best_clf = grid_search.best_estimator_
        results[best_model_name]["model"] = best_clf
        results[best_model_name]["cv"] = f"{grid_search.best_score_:.4f}"
        
        # Re-evaluate on test set
        y_pred = best_clf.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[best_model_name]["acc"] = acc
        results[best_model_name]["prec"] = prec
        results[best_model_name]["rec"] = rec
        results[best_model_name]["f1"] = f1
        
        print(f"  Test F1 after tuning: {f1:.4f}")

    # ── 5. Train Voting Ensemble ───────────────────────────────────────────────
    print("\n── Voting Ensemble (Soft Voting) ─────────────────────────────────")
    
    # Build ensemble from top performing models
    ensemble_estimators = []
    for name, res in results.items():
        clf = res["model"]
        # Wrap LinearSVC for probability support
        if hasattr(clf, 'decision_function'):
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(clf, cv=3)
        ensemble_estimators.append((name, clf))
    
    if len(ensemble_estimators) >= 2:
        voting_clf = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',
            n_jobs=-1
        )
        voting_clf.fit(Xtr, y_train)
        y_pred_ens = voting_clf.predict(Xte)
        
        acc_ens = accuracy_score(y_test, y_pred_ens)
        prec_ens = precision_score(y_test, y_pred_ens, zero_division=0)
        rec_ens = recall_score(y_test, y_pred_ens, zero_division=0)
        f1_ens = f1_score(y_test, y_pred_ens, zero_division=0)
        
        # CV score for ensemble
        try:
            cv_scores_ens = cross_val_score(voting_clf, Xtr, y_train, cv=5, scoring="f1")
            cv_str_ens = f"{cv_scores_ens.mean():.4f} ± {cv_scores_ens.std():.4f}"
        except:
            cv_str_ens = "N/A"
        
        print("{:<25} {:>9.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>15}".format(
            "Voting Ensemble", acc_ens, prec_ens, rec_ens, f1_ens, cv_str_ens))
        
        # Compare with best single model
        best_f1 = results[best_model_name]["f1"]
        if f1_ens > best_f1:
            print(f"  → Ensemble outperforms single model! Using ensemble.")
            best_model_name = "voting_ensemble"
            results["voting_ensemble"] = {
                "model": voting_clf, "acc": acc_ens, "prec": prec_ens, 
                "rec": rec_ens, "f1": f1_ens, "cv": cv_str_ens
            }
        else:
            print(f"  → Single model ({best_model_name}) performs better. Keeping it.")

    # ── 6. Save best model ───────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    best = results[best_model_name]["model"]
    joblib.dump(best, cfg["paths"]["model_path"])
    print(f"\nBest model saved → {cfg['paths']['model_path']}")
    return results, y_test, vec, Xte


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    train_all(cfg)