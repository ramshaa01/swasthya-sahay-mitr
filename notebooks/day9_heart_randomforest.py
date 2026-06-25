import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

os.makedirs('reports/shap', exist_ok=True)
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = joblib.load('data/processed/heart_splits.pkl')

feature_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs']

print("="*60)
print("DAY 9 — Random Forest Heart Disease Tuning")
print("="*60)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print("Note: Small dataset (242 train rows) — aggressive regularization needed")

# ============================================================
# STEP 1: Hyperparameter search
# Small dataset so GridSearch is fast enough here
# Key knobs: constrain max_depth and min_samples to avoid overfitting
# ============================================================
param_dist = {
    'n_estimators':       [50, 100, 200, 300],
    'max_depth':          [2, 3, 4, 5, None],
    'min_samples_split':  [2, 5, 10, 15, 20],
    'min_samples_leaf':   [1, 2, 4, 8],
    'max_features':       ['sqrt', 'log2', None],
    'class_weight':       ['balanced', None]
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=CV,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
search.fit(X_train, y_train)

print(f"\nBest parameters: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ============================================================
# STEP 2: Evaluate best model
# ============================================================
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test ROC-AUC:  {test_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print("\nVS BASELINE:")
print(f"  Baseline Test Accuracy: 0.6557 | Tuned: {test_acc:.4f}")
print(f"  Baseline Test ROC-AUC:  0.7408 | Tuned: {test_auc:.4f}")

# Also run 10-fold CV for a more stable estimate on this small dataset
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv10_auc = cross_val_score(best_model, X_train, y_train,
                            cv=cv10, scoring='roc_auc', n_jobs=-1)
print(f"\n10-fold CV ROC-AUC (more stable on small data): "
      f"{cv10_auc.mean():.4f} (+/- {cv10_auc.std():.4f})")

# ============================================================
# STEP 3: SHAP for heart disease model
# ============================================================
explainer = shap.TreeExplainer(best_model)
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
shap_values = explainer.shap_values(X_test_df)

# shap_values is a list [class0, class1] for Random Forest, or a 3D array (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    sv_class1 = shap_values[:, :, 1]
else:
    sv_class1 = shap_values

mean_shap = pd.Series(
    np.abs(sv_class1).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nGlobal feature importance (mean |SHAP|) for heart disease:")
print(mean_shap)

plt.figure(figsize=(8, 5))
mean_shap.plot(kind='barh', color='steelblue')
plt.xlabel('Mean |SHAP value|')
plt.title('Heart Disease Model — Global Feature Importance (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('reports/shap/heart_shap_global.png')
plt.close()
print("Saved: reports/shap/heart_shap_global.png")

# Single prediction explanation
single_shap = pd.Series(sv_class1[0], index=feature_cols).sort_values()
plt.figure(figsize=(8, 5))
colors = ['red' if v > 0 else 'green' for v in single_shap]
single_shap.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('SHAP value (impact on predicted heart disease risk)')
plt.title('Heart Disease — Single Prediction Explanation (Test Row 0)')
plt.tight_layout()
plt.savefig('reports/shap/heart_shap_single_prediction.png')
plt.close()
print("Saved: reports/shap/heart_shap_single_prediction.png")

joblib.dump(best_model, 'models/tuned_heart_rf.pkl')
joblib.dump(explainer, 'models/explainer_heart.pkl')
print("Saved: models/tuned_heart_rf.pkl")
print("Saved: models/explainer_heart.pkl")

print("\n=== DAY 9 COMPLETE ===")
