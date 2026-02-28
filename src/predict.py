# predict.py
# ======================================================
# Robust Inference / Prediction Engine
# ======================================================

import joblib
import pandas as pd
import numpy as np

from src.utils import clean_data, encode_data, feature_engineering


# ------------------------------------------------------
# REQUIRED FIELDS FOR SINGLE PREDICTION
# ------------------------------------------------------
REQUIRED_FIELDS = [
    "PassengerId", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
]


# ------------------------------------------------------
# Safe default filler (to avoid missing-field errors)
# ------------------------------------------------------
def fill_missing_input(user_input):
    defaults = {
        "PassengerId": 9999,
        "Pclass": 3,
        "Name": "Unknown, Mr. Guest",
        "Sex": "male",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "UNKNOWN",
        "Fare": 20.0,
        "Cabin": "",
        "Embarked": "S"
    }

    final = {}
    for key in REQUIRED_FIELDS:
        final[key] = user_input.get(key, defaults[key])

    return final


# ------------------------------------------------------
# Load model safely
# ------------------------------------------------------
def load_model(model_path):
    return joblib.load(model_path)


# ------------------------------------------------------
# Preprocess single row safely
# ------------------------------------------------------
def preprocess_single(user_input):
    safe_input = fill_missing_input(user_input)
    df = pd.DataFrame([safe_input])

    df = clean_data(df)
    df = encode_data(df)
    df = feature_engineering(df)

    return df


# ------------------------------------------------------
# Predict single passenger survival
# ------------------------------------------------------
def predict_single(model_path, user_input):
    model = load_model(model_path)
    processed = preprocess_single(user_input)

    pred = model.predict(processed)[0]

    return "Survived" if pred == 1 else "Died"


# ------------------------------------------------------
# Batch prediction
# ------------------------------------------------------
def predict_batch(model_path, csv_path):
    model = load_model(model_path)

    raw = pd.read_csv(csv_path)
    raw = clean_data(raw)
    raw = encode_data(raw)
    raw = feature_engineering(raw)

    preds = model.predict(raw)

    raw["Prediction"] = ["Survived" if p == 1 else "Died" for p in preds]

    return raw


# ------------------------------------------------------
# Quick test
# ------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "Fare": 80.0
        # Missing many fields on purpose
    }

    print(predict_single("models/model_default.pkl", sample))