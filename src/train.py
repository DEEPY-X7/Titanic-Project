# train.py
# ======================================================
# Full ML Training Engine
# ======================================================

import joblib
import os

from src.utils import preprocess
from src.model_selection import split_data, select_best_model
from src.tune_model import tune_model


# ------------------------------------------------------
# TRAIN USING DEFAULT BEST MODEL (NO TUNING)
# ------------------------------------------------------
def train_default(input_path, model_save_path="models/model_default.pkl"):
    print("\n[1] Loading + preprocessing data...")
    df = preprocess(input_path)

    print("[2] Selecting best model...")
    best_model_name, best_model = select_best_model(df)

    print(f"[3] Best Model Selected → {best_model_name}")
    print("[4] Saving model...")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_save_path)

    print(f"[✔] Model saved at: {model_save_path}")
    return best_model_name, best_model


# ------------------------------------------------------
# TRAIN WITH HYPERPARAMETER TUNING
# ------------------------------------------------------
def train_with_tuning(input_path, model_name="RandomForest", model_save_path="models/model_tuned.pkl"):
    print("\n[1] Loading + preprocessing data...")
    df = preprocess(input_path)

    print(f"[2] Running hyperparameter tuning for → {model_name}")
    result = tune_model(df, model_name=model_name)

    best_params = result["best_params"]
    best_model = result["best_model"]

    print(f"[3] Best Params Found → {best_params}")
    print("[4] Saving tuned model...")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_save_path)

    print(f"[✔] Tuned model saved at: {model_save_path}")
    return best_params, best_model


# ------------------------------------------------------
# MAIN EXECUTION (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    print("Training default model...")
    train_default("input/data.csv")

    # To train with tuning:
    # train_with_tuning("input/data.csv", model_name="RandomForest")