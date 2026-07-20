"""
model.py
Handles data loading, preprocessing, training, and evaluation of the
intent-classification model used by the Voice Assistant.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os


def load_data(path):
    """Load the labeled intent dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def preprocess_data(data):
    """Basic cleaning: lowercase and strip whitespace from text column."""
    data = data.copy()
    data["text"] = data["text"].astype(str).str.lower().str.strip()
    data = data.dropna(subset=["text", "intent"])
    return data


def train_intent_model(data, model_type="logreg", test_size=0.2, random_state=42):
    """
    Train a TF-IDF + classifier pipeline on the intent dataset.

    model_type: "logreg" (Logistic Regression) or "naive_bayes" (Multinomial NB)
    Returns: (classifier, vectorizer, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["intent"],
        test_size=test_size, random_state=random_state, stratify=data["intent"]
    )

    # NOTE: scikit-learn's default token_pattern (\b\w\w+\b) does not include
    # Unicode combining marks (category Mn), so it incorrectly splits words
    # in scripts that use combining vowel signs -- e.g. Devanagari (Hindi)
    # and Arabic-script (Urdu, Arabic) -- mid-character. Using a
    # whitespace-based token pattern (\S+) keeps whole words intact across
    # every supported language, since all of them use spaces between words.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\S+")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_type == "naive_bayes":
        clf = MultinomialNB()
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return clf, vectorizer, metrics


def save_model(clf, vectorizer, path="trained_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump((clf, vectorizer), f)


def load_model(path="trained_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Quick standalone test
    data = preprocess_data(load_data("data/intents.csv"))
    clf, vec, metrics = train_intent_model(data, "logreg")
    print("Logistic Regression metrics:", metrics)
    clf2, vec2, metrics2 = train_intent_model(data, "naive_bayes")
    print("Naive Bayes metrics:", metrics2)
