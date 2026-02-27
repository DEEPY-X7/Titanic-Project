# tune_model.py
# ======================================================
# Hyperparameter tuning for models
# ======================================================

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.model_selection import split_data


# ------------------------------------------------------
# Logistic Regression Parameter Grid
# ------------------------------------------------------
def logistic_param_grid():
    return {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 5, 10],
        "solver": ["liblinear"]
    }


# ------------------------------------------------------
# RandomForest Parameter Grid
# ------------------------------------------------------
def random_forest_param_grid():
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }


# ------------------------------------------------------
# XGBoost Parameter Grid
# ------------------------------------------------------
def xgb_param_grid():
    return {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }


# ------------------------------------------------------
# Generic Tuning Function
# ------------------------------------------------------
def tune_model(df, model_name="RandomForest"):
    X_train, X_test, y_train, y_test = split_data(df)

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=2000)
        params = logistic_param_grid()

    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        params = xgb_param_grid()

    else:
        model = RandomForestClassifier()
        params = random_forest_param_grid()

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "best_model": grid.best_estimator_
    }