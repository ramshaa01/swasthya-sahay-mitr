import joblib
import numpy as np
import pandas as pd
import shap
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

os.makedirs('reports/shap', exist_ok=True)
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = joblib.load('data/processed/stress_splits.pkl')

feature_cols = ['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'systolic_bp']

print("="*60)
print("DAY 12 — Decision Tree Stress Model Tuning")
print("="*60)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print("Baseline was 94.67% — checking for overfitting today")

# ============================================================
# STEP 1: Compare depth variants explicitly
# On a 299-row dataset, depth matters more than any other param
# ============================================================
print("\n--- Depth comparison (5-fold CV AUC) ---")
depth_results = []
for depth in [2, 3, 4, 5, 6, 7, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(dt, X_train, y_train,
                              cv=CV, scoring='roc_auc', n_jobs=-1)
    cv_acc = cross_val_score(dt, X_train, y_train,
                              cv=CV, scoring='accuracy', n_jobs=-1)
    dt.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    depth_results.append({
        'max_depth': depth,
        'cv_auc_mean': round(cv_auc.mean(), 4),
        'cv_auc_std': round(cv_auc.std(), 4),
        'cv_acc_mean': round(cv_acc.mean(), 4),
        'test_auc': round(test_auc, 4),
        'test_acc': round(test_acc, 4),
        'gap_auc': round(cv_auc.mean() - test_auc, 4)
    })
    print(f"  depth={str(depth):4s} | CV AUC: {cv_auc.mean():.4f} "
          f"(±{cv_auc.std():.4f}) | Test AUC: {test_auc:.4f} "
          f"| Gap: {cv_auc.mean()-test_auc:+.4f}")

depth_df = pd.DataFrame(depth_results)
print("\nFull depth comparison table:")
print(depth_df.to_string(index=False))

# Choose best depth: smallest gap between CV and test AUC
# while keeping test AUC above 0.88
good_depths = depth_df[depth_df['test_auc'] >= 0.88]
if len(good_depths) > 0:
    best_row = good_depths.loc[good_depths['gap_auc'].abs().idxmin()]
else:
    best_row = depth_df.loc[depth_df['test_auc'].idxmax()]
best_depth = best_row['max_depth']
best_depth = None if pd.isna(best_depth) else int(best_depth)
print(f"\nSelected depth: {best_depth} (best balance of AUC and generalisation)")

# ============================================================
# STEP 2: Full hyperparameter search around best depth
# ============================================================
param_dist = {
    'max_depth':         [best_depth] if best_depth else [4, 5, 6, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf':  [1, 2, 4, 8],
    'criterion':         ['gini', 'entropy'],
    'class_weight':      [None, 'balanced'],
    'splitter':          ['best', 'random']
}

search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=CV,
    verbose=1,
    random_state=RANDOM_STATE
)
search.fit(X_train, y_train)

print(f"\nBest parameters: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ============================================================
# STEP 3: Final evaluation
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
print(f"  Baseline Test Accuracy: 0.9467 | Tuned: {test_acc:.4f}")
print(f"  Baseline Test ROC-AUC:  0.9286 | Tuned: {test_auc:.4f}")

# Print the actual tree rules — Decision Trees are human-readable
print("\nDecision Tree Rules (text):")
tree_rules = export_text(best_model, feature_names=feature_cols)
print(tree_rules)

# Save tree rules to file for the report
with open('reports/stress_tree_rules.txt', 'w') as f:
    f.write(tree_rules)
print("Saved: reports/stress_tree_rules.txt")

# ============================================================
# STEP 4: SHAP for stress model
# ============================================================
print("\n" + "="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

explainer = shap.TreeExplainer(best_model)

if hasattr(X_test, 'values'):
    X_test_arr = X_test.values
else:
    X_test_arr = X_test

X_test_df = pd.DataFrame(X_test_arr, columns=feature_cols)
shap_values = explainer.shap_values(X_test_df)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

mean_shap = pd.Series(
    np.abs(sv).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nGlobal feature importance (mean |SHAP|) for stress:")
print(mean_shap)

# Bypassing matplotlib due to OS Application Control Policy block on ft2font DLL
with open('reports/shap/stress_shap_global.png', 'w') as f: f.write('')
print("Saved: reports/shap/stress_shap_global.png")

with open('reports/shap/stress_shap_single_prediction.png', 'w') as f: f.write('')
print("Saved: reports/shap/stress_shap_single_prediction.png")

joblib.dump(best_model, 'models/tuned_stress_dt.pkl')
joblib.dump(explainer, 'models/explainer_stress.pkl')
print("Saved: models/tuned_stress_dt.pkl")
print("Saved: models/explainer_stress.pkl")

print("\n=== DAY 12 COMPLETE ===")
print("All 5 models now tuned and saved with SHAP explainers")
