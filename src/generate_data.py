"""Generate synthetic HR employee data with realistic attrition signals."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 1470

departments = ["Sales", "Research & Development", "Human Resources"]
dept_probs = [0.30, 0.65, 0.05]
roles = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician",
                                "Healthcare Representative", "Manager", "Manufacturing Director"],
    "Human Resources": ["Human Resources", "Manager"],
}
education = ["Below College", "College", "Bachelor", "Master", "Doctor"]
marital = ["Single", "Married", "Divorced"]

rows = []
for eid in range(1, N + 1):
    age = int(np.clip(np.random.normal(37, 9), 18, 60))
    dept = np.random.choice(departments, p=dept_probs)
    role = np.random.choice(roles[dept])
    income = int(np.clip(np.random.lognormal(8.7, 0.4), 1000, 25000))
    distance = int(np.random.exponential(8) + 1)
    overtime = np.random.choice(["Yes", "No"], p=[0.3, 0.7])
    years_at_company = int(np.clip(np.random.exponential(7), 0, 40))
    job_satisfaction = np.random.randint(1, 5)
    wlb = np.random.randint(1, 5)
    edu = np.random.choice(education)
    mar = np.random.choice(marital)

    # Build a logistic-style probability of attrition
    score = -2.0
    if overtime == "Yes": score += 1.5
    if age < 30: score += 0.7
    if income < 3000: score += 0.8
    if distance > 15: score += 0.5
    if job_satisfaction <= 2: score += 0.6
    if wlb <= 2: score += 0.4
    if mar == "Single": score += 0.3
    if years_at_company < 2: score += 0.5
    p = 1 / (1 + np.exp(-score))
    attrition = "Yes" if np.random.random() < p else "No"

    rows.append({
        "EmployeeID": eid,
        "Age": age,
        "Attrition": attrition,
        "Department": dept,
        "JobRole": role,
        "MonthlyIncome": income,
        "DistanceFromHome": distance,
        "OverTime": overtime,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": wlb,
        "Education": edu,
        "MaritalStatus": mar,
        "Gender": np.random.choice(["Male", "Female"], p=[0.6, 0.4]),
        "PercentSalaryHike": np.random.randint(11, 26),
        "TrainingTimesLastYear": np.random.randint(0, 7),
    })

if __name__ == "__main__":
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "data" / "hr_data.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Generated {len(df):,} employees → {out}")
    print(f"   Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
