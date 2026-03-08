# Heart Disease Prediction — KNN Classification

Predicting heart disease risk using K-Nearest Neighbors on 15K+ patient records with 18 health features.

## What It Does

Takes a patient's age, BMI, blood pressure, smoking status, and other health metrics — predicts whether they are at risk of heart disease.

## Pipeline

1. **Cleaning** — Checked nulls, duplicates, verified data types
2. **EDA** — Correlation heatmap, histograms, boxplots, countplots
3. **Encoding** — OrdinalEncoder for ordered features, OneHotEncoder for categorical features
4. **Scaling** — StandardScaler (no outliers), RobustScaler (outlier features) via ColumnTransformer
5. **Modeling** — Pipeline + Elbow Plot (K=1-30), compared Euclidean/Manhattan × Uniform/Distance
6. **Tuning** — GridSearchCV with 120 combinations, 5-Fold Cross-Validation
7. **Evaluation** — Classification Report, Confusion Matrix, Cross-Validation, Overfitting check

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.5% |
| F1 Score (CV) | 87.6% ± 0.4% |
| Overfitting | None (0.9% gap) |
| Best Model | Manhattan, K=29, Uniform |

Picked Manhattan + Uniform — best F1 score across all combinations.

## Quick Start

```
pip install -r requirements.txt
jupyter notebook KNN.ipynb
```

Elton Valiyev — [LinkedIn](https://www.linkedin.com/in/eltonvaliyev/)
