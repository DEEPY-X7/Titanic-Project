# utils.py
# ======================================
# All data preprocessing utilities
# ======================================

import pandas as pd
import numpy as np

# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
def load_data(path):
    """
    Reads CSV and returns raw dataframe.
    """
    return pd.read_csv(path)


# ------------------------------------------------------
# Cleaning functions
# ------------------------------------------------------
def clean_data(df):
    """
    Handles missing values, removes unnecessary columns.
    """
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = df.drop(columns=['Cabin', 'Ticket'])
    df = df.drop_duplicates()

    return df


# ------------------------------------------------------
# Encoding functions
# ------------------------------------------------------
def encode_data(df):
    """
    Converts categorical columns into numeric form.
    """
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].fillna("Other")

    # Convert rare titles into "Other"
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Other')

    return df


# ------------------------------------------------------
# Feature engineering
# ------------------------------------------------------
def feature_engineering(df):
    """
    Adds meaningful features for ML models.
    """
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Alone or not
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], [0, 12, 20, 40, 80], labels=[0, 1, 2, 3])

    # Fare bins
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])

    # Remove unnecessary columns
    df = df.drop(columns=['Name', 'PassengerId'])

    return df


# ------------------------------------------------------
# Full preprocessing pipeline
# ------------------------------------------------------
def preprocess(path):
    """
    Full pipeline:
    load → clean → encode → feature engineering
    Returns final dataframe.
    """
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)
    df = feature_engineering(df)
    df = df.dropna()

    return df