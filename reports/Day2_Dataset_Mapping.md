# Day 2 — Dataset Selection & Feature Mapping

## Chosen Datasets

| Condition | Dataset | Source | Rows (approx) | Target |
|---|---|---|---|---|
| Diabetes | CDC Diabetes Health Indicators (BRFSS 2015) | UCI ML Repo, id=891 (via ucimlrepo) | 253,680 | Diabetes_binary |
| Hypertension | Same BRFSS dataset, reused | UCI ML Repo, id=891 | 253,680 | HighBP |
| Heart Disease | UCI Heart Disease (Cleveland) | UCI ML Repo, id=45 (via ucimlrepo) | 303 | num |
| Obesity | UCI Obesity Levels (eating habits + physical condition) | UCI ML Repo, id=544 (via ucimlrepo) | 2,111 | NObeyesdad |
| Stress | Sleep Health and Lifestyle Dataset | Kaggle: uom190346a/sleep-health-and-lifestyle-dataset | 374 | Stress Level |

## Known Gaps & Decisions Needed
- No dataset includes family history fields. The Digital Twin input form
  can still collect it for context, but no model will train on it unless
  a supplementary dataset is added later.
- The Heart Disease dataset includes clinical-only fields (max heart rate
  during a stress test, ST depression, angiography vessel count) that a
  person without a hospital visit could not realistically self-report.
  Decision needed: use the full feature set and note this as a limitation,
  or restrict the heart disease model to self-reportable fields only
  (age, sex, resting blood pressure, cholesterol, fasting blood sugar).
- Units are inconsistent across datasets for physical activity and blood
  pressure representation. This will be resolved during Day 3 preprocessing.
- No raw glucose value exists in any dataset. The diabetes target is a
  diagnosis label, not a lab value.

## Manual Step (cannot be automated)
Download the Sleep Health and Lifestyle Dataset CSV manually from:
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
and place it in data/stress/
