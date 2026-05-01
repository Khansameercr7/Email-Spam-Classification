"""
evaluate.py
-----------
Generates and saves all evaluation plots:
  • Confusion matrix
  • ROC curve (all models)
  • Precision-Recall curve
  • Feature importance (top spam / ham words)
  • Class distribution bar chart

Run from the project root:  python src/evaluate.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.train import train_all

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
SPAM_COLOR = "#E8593C"
HAM_COLOR  = "#3B8BD4"


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    save_fig(out_path)


# ── 2. ROC Curve ─────────────────────────────────────────────────────────────
def plot_roc_curve(models_dict: dict, Xte, y_test, out_path: str):
    """
    models_dict : {name: sklearn_model}  — only models with predict_proba
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = [SPAM_COLOR, HAM_COLOR, "#27a060", "#7F77DD"]

    for (name, clf), color in zip(models_dict.items(), colors):
        try:
            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(Xte)[:, 1]
            else:
                # calibrate LinearSVC to get probabilities
                cal = CalibratedClassifierCV(clf, cv=3)
                # re-fit is needed here; skipped for brevity — use decision_function
                scores = clf.decision_function(Xte)
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(fontsize=9)
    save_fig(out_path)


# ── 3. Precision-Recall Curve ─────────────────────────────────────────────────
def plot_pr_curve(clf, Xte, y_test, out_path: str):
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(Xte)[:, 1]
    else:
        scores = clf.decision_function(Xte)

    prec, rec, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, color=SPAM_COLOR, lw=2, label=f"PR AUC = {pr_auc:.3f}")
    ax.axhline(y=sum(y_test) / len(y_test), color="gray", linestyle="--",
               label="Baseline (class ratio)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend()
    save_fig(out_path)


# ── 4. Top Features ───────────────────────────────────────────────────────────
def plot_top_features(clf, vectorizer, out_path: str, top_n: int = 20):
    """Works for LogisticRegression (coef_) and NaiveBayes (feature_log_prob_)."""
    try:
        if hasattr(clf, "coef_"):
            coef      = clf.coef_.ravel()
        elif hasattr(clf, "feature_log_prob_"):
            coef      = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        else:
            return

        names     = vectorizer.get_feature_names_out()
        top_spam  = np.argsort(coef)[-top_n:][::-1]
        top_ham   = np.argsort(coef)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].barh([names[i] for i in top_spam[::-1]],
                     coef[top_spam[::-1]], color=SPAM_COLOR, alpha=0.8)
        axes[0].set_title("Top Spam Words", fontsize=12)

        axes[1].barh([names[i] for i in top_ham],
                     -coef[top_ham], color=HAM_COLOR, alpha=0.8)
        axes[1].set_title("Top Ham Words", fontsize=12)

        plt.suptitle("Most Predictive Features", fontsize=13, y=1.02)
        plt.tight_layout()
        save_fig(out_path)
    except Exception as e:
        print(f"  Skipped feature plot: {e}")


# ── 5. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(y_train, y_test, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (y, title) in zip(axes, [(y_train, "Train"), (y_test, "Test")]):
        counts = y.value_counts().sort_index()
        bars   = ax.bar(["Ham", "Spam"], counts.values,
                        color=[HAM_COLOR, SPAM_COLOR], width=0.5, alpha=0.85)
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
        ax.set_title(f"{title} Set Distribution", fontsize=12)
        ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig(out_path)


# ── 6. Word Cloud for Spam Emails ───────────────────────────────────────────
def plot_wordcloud(df, text_col, label_col, out_path: str):
    """Generate word cloud for spam emails."""
    spam_text = " ".join(df[df[label_col] == 1][text_col].astype(str))
    
    if not spam_text.strip():
        print(f"  Skipped wordcloud: no spam emails found")
        return
    
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Reds",
        max_words=100,
        stopwords=None
    ).generate(spam_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Spam Emails", fontsize=13)
    save_fig(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    rd  = cfg["paths"]["reports_dir"]
    os.makedirs(rd, exist_ok=True)

    print("Training models …")
    results, y_test, vectorizer, Xte = train_all(cfg)

    best_name = cfg["training"]["best_model"]
    best_clf  = results[best_name]["model"]
    y_pred    = best_clf.predict(Xte)

    # ── Load y_train for distribution plot ────────────────────────────────────
    y_train = pd.read_csv(cfg["data"]["train_path"])["spam"]

    print("\nGenerating evaluation plots …")
    plot_confusion_matrix(y_test, y_pred,         f"{rd}confusion_matrix.png")
    plot_top_features(best_clf, vectorizer,        f"{rd}top_features.png")
    plot_class_distribution(y_train, y_test,       f"{rd}class_distribution.png")
    
    # Load raw data for word cloud
    raw_df = pd.read_csv(cfg["data"]["raw_path"])
    plot_wordcloud(raw_df, cfg["data"]["text_column"], cfg["data"]["label_column"], 
                   f"{rd}wordcloud_spam.png")

    models_for_roc = {name: r["model"] for name, r in results.items()}
    plot_roc_curve(models_for_roc, Xte, y_test,   f"{rd}roc_curve.png")
    plot_pr_curve(best_clf, Xte, y_test,           f"{rd}precision_recall_curve.png")

    # ── Classification report ─────────────────────────────────────────────────
    print("\n── Classification Report ─────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))