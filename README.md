# 👥 HR Analytics – Employee Attrition Analysis

Predictive analytics on employee turnover. Identify what factors drive attrition and build a model to flag at-risk employees.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

Employee turnover is expensive — recruiting, training, and lost productivity. This project uses descriptive and predictive analytics to:

1. Understand **who leaves** and **why**
2. Build a model to **predict attrition** risk
3. Translate findings into **HR retention strategies**

## ❓ Business Questions

- What's the company-wide attrition rate?
- Which departments have the highest turnover?
- Do salary, overtime, or commute distance predict who leaves?
- Are younger employees more likely to leave?
- What's the role of job satisfaction & work-life balance?

## 📁 Structure

```
hr-analytics/
├── data/
│   └── hr_data.csv
├── notebooks/
│   └── attrition_analysis.ipynb
├── src/
│   ├── eda.py
│   ├── model.py
│   └── generate_data.py
├── outputs/
├── images/
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

Python · Pandas · Scikit-learn · Matplotlib · Seaborn

## 🚀 Run

```bash
git clone https://github.com/yourusername/hr-analytics.git
cd hr-analytics
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py
python src/model.py
```

## 📊 Key Findings

- 📉 Overall attrition rate: **~16%**
- 🌙 Employees who work **overtime** are **3x more likely** to leave
- 💼 **Sales department** has highest turnover
- 🎓 Single, young (<30), early-career employees are highest risk
- 🧠 Top predictors: OverTime, MonthlyIncome, Age, JobSatisfaction

## 🤖 Model Performance

| Model | Accuracy | F1 (Attrition class) |
|-------|----------|----------------------|
| Logistic Regression | 87% | 0.51 |
| Random Forest | 89% | 0.56 |

## 📝 License

[MIT](LICENSE)
