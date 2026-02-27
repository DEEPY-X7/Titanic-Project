# model_selection.py
# ======================================================
# Handle model comparison, scoring, and selection
# ======================================================

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ------------------------------------------------------
# Function: train-test split
# ------------------------------------------------------
def split_data(df, target='Survived'):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------
# Function: define candidate models
# ------------------------------------------------------
def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    return models


# ------------------------------------------------------
# Function: evaluate a model
# ------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }


# ------------------------------------------------------
# Function: compare all models
# ------------------------------------------------------
def compare_models(df):
    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()
    results = {}

    for name, model in models.items():
        scores = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = scores

    return results


# ------------------------------------------------------
# Function: select best model based on F1 Score
# ------------------------------------------------------
def select_best_model(df):
    results = compare_models(df)

    # Find model with highest F1 score
    best_model_name = max(results, key=lambda m: results[m]["f1"])

    # Instantiate fresh model
    model = get_models()[best_model_name]

    # Final training on full dataset
    X_train, X_test, y_train, y_test = split_data(df)
    model.fit(X_train, y_train)

    return best_model_name, model