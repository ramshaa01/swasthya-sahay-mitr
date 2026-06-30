# Swasthya Sahay Mitr — ML Phase Summary
### Week 2 Complete (Days 8-13)

## Final Model Results

| Condition | Model | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Diabetes | XGBoost | 70.77% | 0.7985 |
| Hypertension | Logistic Regression | 71.70% | 0.7933 |
| Heart Disease | Random Forest | 67.21% | 0.7121 |
| Obesity | Gradient Boosting | 92.91% | 0.9690 |
| Stress | Decision Tree | 98.67% | 0.9954 |

## Key Decisions & Justifications

### Why these 5 models?
- XGBoost (Diabetes): handles non-linear interactions between BMI, age,
  and lifestyle inputs that simpler models miss.
- Logistic Regression (Hypertension): the lifestyle-to-BP relationship
  approximates linearity; LR is interpretable and its coefficients
  directly validate SHAP rankings.
- Random Forest (Heart Disease): robust to correlated inputs; handles
  the small Cleveland dataset (303 rows) better than gradient boosting.
- Gradient Boosting (Obesity): captures diet x activity interaction
  effects; achieved 0.969 AUC on lifestyle features alone after
  data leakage was corrected.
- Decision Tree (Stress): shallow tree (depth 4) produces directly
  human-readable rules; 0.995 AUC confirms sleep quality and gender
  as dominant stress predictors.

### Key findings from SHAP analysis
- Diabetes: GenHlth (general self-assessed health) was the strongest
  predictor, outweighing BMI — consistent with EDA correlations.
- Hypertension: Age dominated by a wide margin (SHAP 0.611 vs 0.378
  for second-place HighChol) — aligns with medical literature.
- Heart Disease: Sex and age dominated; cholesterol and fasting blood
  sugar showed near-zero linear correlation but retained some SHAP
  contribution via non-linear interactions.
- Obesity: family_history_with_overweight ranked #1 (SHAP 1.205),
  with snacking patterns (CAEC) and meal frequency (NCP) outranking
  physical activity — diet patterns are stronger predictors than exercise.
- Stress: Quality of Sleep was #1 (SHAP 0.284); the tree rules show
  that females with sleep quality <= 6.5 are directly classified as
  high stress — clinically intuitive.

### Data leakage discovered and fixed
The obesity model initially achieved 99.53% accuracy because BMI,
Height, and Weight were included as features — the model was computing
the obesity definition rather than learning lifestyle patterns. These
three features were removed on Day 7; honest accuracy is 92.91%.

### Acknowledged limitations
- Heart disease model uses only 5 self-reportable features (242 training
  rows) to stay consistent with the no-clinic premise; this constrains
  AUC to 0.71.
- Diabetes and hypertension AUC (~0.79-0.80) reflect the ceiling of
  self-reported lifestyle data without clinical measurements (consistent
  with published BRFSS-dataset benchmarks of 0.78-0.83).
- Stress and obesity datasets are small (374 and 2111 rows respectively);
  results should be validated on larger cohorts in future work.

## Files Ready for Backend (Week 3)

| File | Purpose |
|---|---|
| models/tuned_diabetes_xgboost.pkl | Diabetes inference |
| models/tuned_hypertension_lr.pkl | Hypertension inference |
| models/tuned_heart_rf.pkl | Heart disease inference |
| models/tuned_obesity_gb.pkl | Obesity inference |
| models/tuned_stress_dt.pkl | Stress inference |
| models/explainer_diabetes.pkl | SHAP for diabetes |
| models/explainer_hypertension.pkl | SHAP for hypertension |
| models/explainer_heart.pkl | SHAP for heart disease |
| models/explainer_obesity.pkl | SHAP for obesity |
| models/explainer_stress.pkl | SHAP for stress |
| models/scaler_hypertension.pkl | Scaling for LR inputs |
| models/obesity_feature_cols.pkl | Feature order for obesity |
| models/models_config.pkl | Unified config for backend |
