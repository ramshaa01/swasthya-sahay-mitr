import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('reports/shap', exist_ok=True)

# ============================================================
# CONFIGURATION — maps each condition to its model, explainer,
# feature list, and the processed splits it was trained on.
# This same config dict will be copy-pasted into the backend
# on Day 17, so getting it right here matters.
# ============================================================

MODELS_CONFIG = {
    'diabetes': {
        'model_path':    'models/tuned_diabetes_xgboost.pkl',
        'explainer_path':'models/explainer_diabetes.pkl',
        'splits_path':   'data/processed/diabetes_splits.pkl',
        'feature_cols':  ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
                          'HighChol', 'CholCheck', 'PhysActivity', 'Fruits',
                          'Veggies', 'HvyAlcoholConsump', 'Smoker', 'Sex',
                          'DiffWalk', 'HighBP'],
        'target':        'Diabetes_binary',
        'model_type':    'tree'
    },
    'hypertension': {
        'model_path':    'models/tuned_hypertension_lr.pkl',
        'explainer_path':'models/explainer_hypertension.pkl',
        'splits_path':   'data/processed/hypertension_splits.pkl',
        'scaler_path':   'models/scaler_hypertension.pkl',
        'feature_cols':  ['Age', 'BMI', 'GenHlth', 'HighChol', 'Smoker',
                          'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk',
                          'MentHlth', 'PhysHlth', 'Sex'],
        'target':        'HighBP',
        'model_type':    'linear'
    },
    'heart': {
        'model_path':    'models/tuned_heart_rf.pkl',
        'explainer_path':'models/explainer_heart.pkl',
        'splits_path':   'data/processed/heart_splits.pkl',
        'feature_cols':  ['age', 'sex', 'trestbps', 'chol', 'fbs'],
        'target':        'heart_disease_binary',
        'model_type':    'tree'
    },
    'obesity': {
        'model_path':    'models/tuned_obesity_gb.pkl',
        'explainer_path':'models/explainer_obesity.pkl',
        'splits_path':   'data/processed/obesity_splits.pkl',
        'feature_cols':  None,  # loaded from models/obesity_feature_cols.pkl
        'target':        'obesity_risk',
        'model_type':    'tree'
    },
    'stress': {
        'model_path':    'models/tuned_stress_dt.pkl',
        'explainer_path':'models/explainer_stress.pkl',
        'splits_path':   'data/processed/stress_splits.pkl',
        'feature_cols':  ['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
                          'Physical Activity Level', 'systolic_bp'],
        'target':        'high_stress',
        'model_type':    'tree'
    }
}

# Load obesity feature cols from saved pkl
MODELS_CONFIG['obesity']['feature_cols'] = joblib.load('models/obesity_feature_cols.pkl')

# ============================================================
# STEP 1: Verify every model + explainer loads correctly
# and produces valid SHAP output
# ============================================================
print("="*60)
print("STEP 1: Verifying all models and explainers")
print("="*60)

all_results = []

for condition, config in MODELS_CONFIG.items():
    print(f"\n--- {condition.upper()} ---")
    model = joblib.load(config['model_path'])
    explainer = joblib.load(config['explainer_path'])
    X_train, X_test, y_train, y_test = joblib.load(config['splits_path'])
    feature_cols = config['feature_cols']

    # Get a single test row as DataFrame (same format backend will use)
    if hasattr(X_test, 'values'):
        X_sample = X_test[:5]
    else:
        X_sample = X_test[:5]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)

    # Test prediction
    prob = model.predict_proba(X_sample_df)[:, 1]
    print(f"  Prediction (5 rows): {np.round(prob, 3)}")

    # Test SHAP
    shap_vals = explainer.shap_values(X_sample_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif len(np.array(shap_vals).shape) == 3:
        shap_vals = shap_vals[:, :, 1]
        
    print(f"  SHAP shape: {np.array(shap_vals).shape} — OK")

    # Top 3 SHAP features for first row
    row_shap = pd.Series(shap_vals[0], index=feature_cols).abs().sort_values(ascending=False)
    print(f"  Top 3 features for row 0: {row_shap.head(3).index.tolist()}")
    print(f"  Model loaded: {type(model).__name__} OK")
    print(f"  Explainer loaded: {type(explainer).__name__} OK")


# ============================================================
# STEP 2: Unified SHAP summary — top feature per condition
# ============================================================
print("\n" + "="*60)
print("STEP 2: Global SHAP summary across all 5 models")
print("="*60)

summary_rows = []
for condition, config in MODELS_CONFIG.items():
    model = joblib.load(config['model_path'])
    explainer = joblib.load(config['explainer_path'])
    X_train, X_test, y_train, y_test = joblib.load(config['splits_path'])
    feature_cols = config['feature_cols']

    sample_size = min(200, X_test.shape[0])
    X_sample = X_test[:sample_size]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)

    shap_vals = explainer.shap_values(X_sample_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif len(np.array(shap_vals).shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_cols)
    top3 = mean_shap.sort_values(ascending=False).head(3)
    summary_rows.append({
        'condition': condition,
        'top_feature_1': top3.index[0],
        'top_feature_2': top3.index[1],
        'top_feature_3': top3.index[2],
        'top_shap_1': round(top3.iloc[0], 4),
        'top_shap_2': round(top3.iloc[1], 4),
        'top_shap_3': round(top3.iloc[2], 4),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv('reports/shap_summary_all_models.csv', index=False)
print("\nSaved: reports/shap_summary_all_models.csv")


# ============================================================
# STEP 3: Final model results table for the report
# ============================================================
print("\n" + "="*60)
print("STEP 3: Final results table")
print("="*60)

from sklearn.metrics import accuracy_score, roc_auc_score

final_results = []
model_names = {
    'diabetes':     'XGBoost',
    'hypertension': 'Logistic Regression',
    'heart':        'Random Forest',
    'obesity':      'Gradient Boosting',
    'stress':       'Decision Tree'
}

for condition, config in MODELS_CONFIG.items():
    model = joblib.load(config['model_path'])
    _, X_test, _, y_test = joblib.load(config['splits_path'])
    feature_cols = config['feature_cols']
    X_test_df = pd.DataFrame(
        X_test.values if hasattr(X_test, 'values') else X_test,
        columns=feature_cols
    )
    y_prob = model.predict_proba(X_test_df)[:, 1]
    y_pred = model.predict(X_test_df)
    final_results.append({
        'condition':    condition,
        'model':        model_names[condition],
        'test_accuracy': round(accuracy_score(y_test, y_pred), 4),
        'test_roc_auc':  round(roc_auc_score(y_test, y_prob), 4)
    })

results_df = pd.DataFrame(final_results)
print(results_df.to_string(index=False))
results_df.to_csv('reports/final_model_results.csv', index=False)
print("\nSaved: reports/final_model_results.csv")

# Save the full MODELS_CONFIG for backend import on Day 17
joblib.dump(MODELS_CONFIG, 'models/models_config.pkl')
print("Saved: models/models_config.pkl")

print("\n=== DAY 13 COMPLETE ===")
print("All 5 models verified, SHAP working, config saved for backend")
