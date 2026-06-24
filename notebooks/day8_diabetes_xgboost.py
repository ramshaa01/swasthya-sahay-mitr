import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

os.makedirs('models', exist_ok=True)
os.makedirs('reports/shap', exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Load preprocessed splits
X_train, X_test, y_train, y_test = joblib.load('data/processed/diabetes_splits.pkl')

feature_cols = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
                'HighChol', 'CholCheck', 'PhysActivity', 'Fruits',
                'Veggies', 'HvyAlcoholConsump', 'Smoker', 'Sex',
                'DiffWalk', 'HighBP']

print("="*60)
print("DAY 8 — XGBoost Diabetes Tuning")
print("="*60)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ============================================================
# STEP 1: Hyperparameter search
# Large dataset so RandomizedSearchCV with 10 iterations is
# enough to find a good region without taking all day
# ============================================================
param_dist = {
    'n_estimators':   [100, 200, 300, 500],
    'max_depth':      [3, 4, 5, 6],
    'learning_rate':  [0.01, 0.05, 0.1, 0.2],
    'subsample':      [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma':          [0, 0.1, 0.2]
}

base_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nRunning RandomizedSearchCV (10 iterations, scoring=roc_auc)...")
print("This may take 5-10 minutes due to large training set size...")

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=CV,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
search.fit(X_train, y_train)

print(f"\nBest parameters found: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ============================================================
# STEP 2: Train final model with best params and evaluate
# ============================================================
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test ROC-AUC:  {test_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Compare against baseline
print("\nVS BASELINE:")
print(f"  Baseline Test Accuracy: 0.7040 | Tuned: {test_acc:.4f}")
print(f"  Baseline Test ROC-AUC:  0.8017 | Tuned: {test_auc:.4f}")

# Save the tuned model
joblib.dump(best_model, 'models/tuned_diabetes_xgboost.pkl')
print("\nSaved: models/tuned_diabetes_xgboost.pkl")

# ============================================================
# STEP 3: SHAP integration — first test on diabetes model
# ============================================================
print("\n" + "="*60)
print("SHAP EXPLAINABILITY TEST")
print("="*60)

# Use TreeExplainer for XGBoost — fast and exact
explainer = shap.TreeExplainer(best_model)

# Compute SHAP on a sample of 500 test rows (full test set is slow)
sample_size = min(500, X_test.shape[0])
if hasattr(X_test, 'values'):
    X_sample = X_test.values[:sample_size]
else:
    X_sample = X_test[:sample_size]

X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)
shap_values = explainer.shap_values(X_sample_df)

print(f"SHAP values shape: {np.array(shap_values).shape}")
print("SHAP computation successful!")

# Mean absolute SHAP per feature (global importance)
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nGlobal feature importance (mean |SHAP|):")
print(mean_shap)

# Bar chart of global SHAP importance
plt.figure(figsize=(10, 6))
mean_shap.plot(kind='barh', color='steelblue')
plt.xlabel('Mean |SHAP value|')
plt.title('Diabetes Model — Global Feature Importance (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('reports/shap/diabetes_shap_global.png')
plt.close()
print("Saved: reports/shap/diabetes_shap_global.png")

# Per-prediction SHAP bar chart for a single example (row 0 of test set)
print("\nSingle-prediction SHAP example (test row 0):")
single_shap = pd.Series(shap_values[0], index=feature_cols).sort_values()
print(single_shap)

plt.figure(figsize=(10, 6))
colors = ['red' if v > 0 else 'green' for v in single_shap]
single_shap.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('SHAP value (impact on predicted diabetes risk)')
plt.title('Diabetes — Single Prediction Explanation (Test Row 0)')
plt.tight_layout()
plt.savefig('reports/shap/diabetes_shap_single_prediction.png')
plt.close()
print("Saved: reports/shap/diabetes_shap_single_prediction.png")

# Save the explainer alongside the model
joblib.dump(explainer, 'models/explainer_diabetes.pkl')
print("Saved: models/explainer_diabetes.pkl")

print("\n=== DAY 8 COMPLETE ===")
print("Tuned XGBoost model + SHAP explainer saved and ready for backend")
