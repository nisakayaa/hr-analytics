"""
HR Attrition – Predictive Modeling
===================================
Trains and evaluates Logistic Regression + Random Forest for attrition prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def prepare_features(df: pd.DataFrame):
    """Encode categoricals & scale numerics."""
    df = df.copy()
    target = (df["Attrition"] == "Yes").astype(int)
    df = df.drop(columns=["Attrition", "EmployeeID", "AttritionFlag"], errors="ignore")

    # Label-encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    return df, target


def train_models(X, y):
    """Train logistic regression and random forest, return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    proba_lr = lr.predict_proba(X_test_s)[:, 1]
    results["LogisticRegression"] = {
        "report": classification_report(y_test, pred_lr, output_dict=True),
        "auc": roc_auc_score(y_test, proba_lr),
        "model": lr,
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)  # tree-based doesn't need scaling
    pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = {
        "report": classification_report(y_test, pred_rf, output_dict=True),
        "auc": roc_auc_score(y_test, proba_rf),
        "model": rf,
        "feature_importance": pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False),
    }

    return results


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "hr_data.csv")
    X, y = prepare_features(df)
    results = train_models(X, y)

    for name, r in results.items():
        print(f"\n📊 {name}")
        print(f"   AUC: {r['auc']:.3f}")
        print(f"   Accuracy: {r['report']['accuracy']:.3f}")
        print(f"   Attrition F1: {r['report']['1']['f1-score']:.3f}")

    print("\n🔝 TOP 10 FEATURES (Random Forest):")
    print(results["RandomForest"]["feature_importance"].head(10))


if __name__ == "__main__":
    main()
