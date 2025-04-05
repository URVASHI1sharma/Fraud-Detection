# 🛡️ Fraud Detection in Banking Transactions
This project focuses on detecting fraudulent transactions using machine learning models in Python. It is part of a larger portfolio demonstrating skills in data science, machine learning, and cybersecurity within the finance domain.

## 📌 Problem Statement
Financial institutions lose billions every year due to fraudulent transactions. This project aims to build a predictive model that identifies potentially fraudulent banking transactions, thereby helping institutions to flag and prevent them in real time.

## 📊 Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset]
- **Description**: Contains anonymized features of transactions (`V1` to `V28`), `Time`, `Amount`, and the `Class` label (0 for legit, 1 for fraud).
- **Size**: 284,807 transactions with 492 frauds (highly imbalanced dataset)

## 🛠️ Technologies Used
- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Pickle (for model serialization)

## 🧪 Project Workflow

### 1. Data Preprocessing
- Handled class imbalance using Stratified Shuffle Split.
- Scaled features with `StandardScaler`.
- Split into training and testing datasets.

### 2. Exploratory Data Analysis (EDA)
- Analyzed fraud vs non-fraud distribution.
- Visualized transaction amounts and correlation matrix.

### 3. Feature Engineering
- Created new features for deeper analysis (if any).
- Handled skewed distributions and outliers.

### 4. Model Building & Evaluation
Trained and compared three models:

| Model                | Precision | Recall | F1 Score | ROC AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.40      | 0.89   | 0.55     | 0.98    |
| Random Forest       | 0.96      | 0.67   | 0.79     | 0.97    |
| XGBoost             | 0.88      | 0.84   | 0.86     | 0.99 ✅ |

✅ **XGBoost** was selected as the best model based on high F1-score and ROC AUC.

## ✅ Final Model

- Selected Model: **XGBoost Classifier**
- Optimized for threshold = `0.9` for better precision.
- Serialized using `pickle` for deployment and reuse.

```python
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
