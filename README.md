# HR Analytics: Employee Attrition

Predicts whether an employee leaves, using logistic regression and a random forest, on 1,470 synthetic employee records. The generator is the interesting part: it builds each person's attrition probability from a logistic score I wrote by hand, weighting overtime, low income, long commute, low satisfaction and short tenure. So there is a known right answer, and the question becomes whether the models find it.

## What came out

Attrition runs at 37.8% in the generated data. Logistic regression gets AUC 0.677 and F1 0.568 on the leaving class. The random forest gets a better AUC at 0.702 but a worse F1 at 0.537, so which model is "better" depends on whether you care about ranking risk or about catching leavers.

Overtime is the clearest single signal in the EDA: 61.5% of people working overtime leave, against 28.7% of those who don't. By age band, the 18-25 group leaves most at 50.9%.

The random forest's feature importances tell a different story from the EDA. Its top three are MonthlyIncome, Age and DistanceFromHome, with OverTime only sixth, even though overtime carries the largest coefficient in my generator. Continuous features with many split points crowd out a binary one. That gap between what I planted and what the importances report is the main thing this project taught me.

## Running it

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py
python src/model.py
```

On Windows, set `PYTHONIOENCODING=utf-8` first or the emoji in the print statements will raise `UnicodeEncodeError`.

Both models use `class_weight="balanced"`, and categorical columns are label-encoded. One-hot would be more correct for the unordered ones like JobRole, and I'd add SHAP values if I came back to this.

Built with pandas, numpy, scikit-learn, matplotlib and seaborn.
