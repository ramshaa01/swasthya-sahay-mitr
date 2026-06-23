import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Train, cross-validate, and evaluate a model. Print and return results."""
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV,
                                scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    cv_auc = cross_val_score(model, X_train, y_train, cv=CV,
                             scoring='roc_auc', n_jobs=-1)
    print(f"CV ROC-AUC:   {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    # Fit on full training set and evaluate on held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return model, {
        'name': name,
        'cv_accuracy': round(cv_scores.mean(), 4),
        'cv_accuracy_std': round(cv_scores.std(), 4),
        'cv_roc_auc': round(cv_auc.mean(), 4),
        'test_accuracy': round(test_acc, 4),
        'test_roc_auc': round(test_auc, 4)
    }


results = []

# ============================================================
# 1. DIABETES — XGBoost baseline
# ============================================================
X_train, X_test, y_train, y_test = joblib.load('data/processed/diabetes_splits.pkl')
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE
)
fitted_model, res = evaluate_model(model, X_train, X_test, y_train, y_test, 'Diabetes (XGBoost)')
results.append(res)
joblib.dump(fitted_model, 'models/baseline_diabetes_xgboost.pkl')
print("Saved: models/baseline_diabetes_xgboost.pkl")


# ============================================================
# 2. HYPERTENSION — Logistic Regression baseline (scaled inputs)
# ============================================================
X_train, X_test, y_train, y_test = joblib.load('data/processed/hypertension_splits.pkl')
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
fitted_model, res = evaluate_model(model, X_train, X_test, y_train, y_test,
                                   'Hypertension (Logistic Regression)')
results.append(res)
joblib.dump(fitted_model, 'models/baseline_hypertension_lr.pkl')
print("Saved: models/baseline_hypertension_lr.pkl")


# ============================================================
# 3. HEART DISEASE — Random Forest baseline
# ============================================================
X_train, X_test, y_train, y_test = joblib.load('data/processed/heart_splits.pkl')
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
fitted_model, res = evaluate_model(model, X_train, X_test, y_train, y_test,
                                   'Heart Disease (Random Forest)')
results.append(res)
joblib.dump(fitted_model, 'models/baseline_heart_rf.pkl')
print("Saved: models/baseline_heart_rf.pkl")


# ============================================================
# 4. OBESITY — Gradient Boosting baseline
# ============================================================
X_train, X_test, y_train, y_test = joblib.load('data/processed/obesity_splits.pkl')
model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, random_state=RANDOM_STATE)
fitted_model, res = evaluate_model(model, X_train, X_test, y_train, y_test,
                                   'Obesity (Gradient Boosting)')
results.append(res)
joblib.dump(fitted_model, 'models/baseline_obesity_gb.pkl')
print("Saved: models/baseline_obesity_gb.pkl")


# ============================================================
# 5. STRESS — Decision Tree baseline (shallow, max_depth=5)
# ============================================================
X_train, X_test, y_train, y_test = joblib.load('data/processed/stress_splits.pkl')
model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
fitted_model, res = evaluate_model(model, X_train, X_test, y_train, y_test,
                                   'Stress (Decision Tree)')
results.append(res)
joblib.dump(fitted_model, 'models/baseline_stress_dt.pkl')
print("Saved: models/baseline_stress_dt.pkl")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("BASELINE RESULTS SUMMARY")
print("="*60)
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
summary_df.to_csv('reports/baseline_results.csv', index=False)
print("\nSaved: reports/baseline_results.csv")
print("\nFlag: any model below 0.75 accuracy needs attention on Day 8-12")
print("Flag: heart disease accuracy may be lower due to only 242 training rows")
