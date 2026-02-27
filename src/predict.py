# predict.py
# ======================================================
# Inference / Prediction Engine
# ======================================================

import joblib
import pandas as pd
import numpy as np

from src.utils import clean_data, encode_data, feature_engineering


# ------------------------------------------------------
# Load model
# ------------------------------------------------------
def load_model(model_path):
    """
    Loads a saved .pkl model.
    """
    return joblib.load(model_path)


# ------------------------------------------------------
# Preprocess a single row of new data
# ------------------------------------------------------
def preprocess_single(input_dict):
    """
    Accepts a python dictionary with passenger features.
    Converts it into a model-ready row (DataFrame).
    """
    df = pd.DataFrame([input_dict])

    # Follow the same cleaning steps
    df = clean_data(df)
    df = encode_data(df)
    df = feature_engineering(df)

    return df


# ------------------------------------------------------
# Predict for a single passenger
# ------------------------------------------------------
def predict_single(model_path, user_input):
    """
    Predicts survival for one passenger.
    user_input must be a dictionary.
    """
    model = load_model(model_path)
    processed = preprocess_single(user_input)

    prediction = model.predict(processed)[0]

    return "Survived" if prediction == 1 else "Died"


# ------------------------------------------------------
# Batch prediction for multiple rows
# ------------------------------------------------------
def predict_batch(model_path, csv_path):
    """
    Takes a CSV file, preprocesses it, predicts for all rows.
    """
    model = load_model(model_path)

    # Load raw CSV
    df = pd.read_csv(csv_path)

    # Run preprocessing pipeline (manual because preprocess() expects PassengerId etc.)
    df = clean_data(df)
    df = encode_data(df)
    df = feature_engineering(df)

    preds = model.predict(df)

    df["Prediction"] = ["Survived" if p == 1 else "Died" for p in preds]

    return df


# ------------------------------------------------------
# CLI runner
# ------------------------------------------------------
if __name__ == "__main__":
    example = {
        "PassengerId": 1000,
        "Pclass": 1,
        "Name": "Allen, Mr. John",
        "Sex": "male",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5",
        "Fare": 70.0,
        "Cabin": "",
        "Embarked": "S"
    }

    print(predict_single("models/model_default.pkl", example))