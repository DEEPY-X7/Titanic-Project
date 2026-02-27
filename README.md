# 🛳 Titanic Survival Prediction — ML Production System

This project is a fully modular, production-grade Machine Learning system for predicting Titanic passenger survival.  
It follows best practices used in real companies: modular code, clean architecture, hyperparameter tuning, and inference engine.

---

## 📁 Project Structure  
```
Titanic-Project/
│
├── input/
│   └── data.csv
│
├── src/
│   ├── utils.py
│   ├── model_selection.py
│   ├── tune_model.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model_default.pkl
│   ├── model_tuned.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
└── README.md
```

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

(Or install manually: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, joblib)

---

## 🔄 Full Data Preprocessing Pipeline

### Steps:
1. Load dataset  
2. Clean missing values  
3. Encode categorical features  
4. Feature engineering  
   - Title extraction  
   - Family size  
   - IsAlone  
   - Age bins  
   - Fare bins  

All preprocessing in:

```
src/utils.py
```

Run full preprocessing anywhere:

```python
from src.utils import preprocess
df = preprocess("input/data.csv")
```

---

## 🤖 Model Selection & Evaluation

Automatic evaluation of:
- Logistic Regression  
- Random Forest  
- XGBoost  

Functions available in:

```
src/model_selection.py
```

Run comparison:

```python
from src.model_selection import compare_models
compare_models(df)
```

---

## 🔍 Hyperparameter Tuning

GridSearchCV tuning for:
- Logistic Regression  
- Random Forest  
- XGBoost  

Run tuning:

```python
from src.tune_model import tune_model
tune_model(df, model_name="RandomForest")
```

---

## 🏋️ Training Engine

### Train best model (no tuning):

```
python src/train.py
```

### Train with hyperparameter tuning:

```python
from src.train import train_with_tuning
train_with_tuning("input/data.csv", model_name="XGBoost")
```

Trained models saved to:

```
models/
```

---

## 🧠 Prediction / Inference

### Single prediction:

```python
from src.predict import predict_single

user = {
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

predict_single("models/model_default.pkl", user)
```

### Batch prediction:

```python
from src.predict import predict_batch
predict_batch("models/model_default.pkl", "input/data.csv")
```

---

## 📊 Notebook Exploration

All EDA and visual analysis done inside:

```
notebooks/exploration.ipynb
```

---

## 🏁 Final Notes

This project is designed for:
- ML engineers  
- Data scientists  
- Students  
- Kaggle competitors  
- Production deployments (API ready)

Modular, reusable, and scalable.
