"""
HR Attrition – Exploratory Data Analysis
========================================
Surfaces relationships between employee features and attrition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    print(f"✅ Loaded {len(df):,} employees")
    print(f"   Attrition rate: {df['AttritionFlag'].mean():.2%}")
    return df


def attrition_by_department(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Department")["AttritionFlag"].mean().sort_values(ascending=False)


def attrition_by_overtime(df: pd.DataFrame) -> pd.Series:
    return df.groupby("OverTime")["AttritionFlag"].mean()


def attrition_by_age_band(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["AgeBand"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
    return df.groupby("AgeBand", observed=True)["AttritionFlag"].mean()


def income_distribution(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df, x="MonthlyIncome", hue="Attrition", fill=True, ax=ax)
    ax.set_title("Monthly Income Distribution by Attrition", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, annot=False, cbar_kws={"shrink": .7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = load(str(root / "data" / "hr_data.csv"))

    print("\n🏢 ATTRITION BY DEPARTMENT:")
    print(attrition_by_department(df).map("{:.2%}".format))

    print("\n🌙 ATTRITION BY OVERTIME:")
    print(attrition_by_overtime(df).map("{:.2%}".format))

    print("\n🎂 ATTRITION BY AGE BAND:")
    print(attrition_by_age_band(df).map("{:.2%}".format))

    images = root / "images"
    images.mkdir(exist_ok=True)
    income_distribution(df, str(images / "income_dist.png"))
    correlation_heatmap(df, str(images / "corr_heatmap.png"))


if __name__ == "__main__":
    main()
