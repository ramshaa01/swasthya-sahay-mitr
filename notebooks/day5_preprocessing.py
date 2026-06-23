import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_and_smote(X, y, dataset_name, use_smote=True):
    """Split into train/test, apply SMOTE on training data only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n{dataset_name} — Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Train class balance before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Train class balance after SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("SMOTE skipped (classes already balanced)")

    return X_train_res, X_test, y_train_res, y_test


def scale_features(X_train, X_test, dataset_name):
    """Fit scaler on training data only, transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f'models/scaler_{dataset_name}.pkl')
    print(f"Scaler saved: models/scaler_{dataset_name}.pkl")
    return X_train_scaled, X_test_scaled


def save_splits(X_train, X_test, y_train, y_test, name):
    """Save processed splits to data/processed/ for use in model training."""
    joblib.dump((X_train, X_test, y_train, y_test), f'data/processed/{name}_splits.pkl')
    print(f"Splits saved: data/processed/{name}_splits.pkl")


# ============================================================
# 1. DIABETES — XGBoost (tree-based, no scaling needed)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING: DIABETES")
print("="*60)
df = pd.read_csv('data/diabetes/cleaned_diabetes_brfss.csv')
feature_cols = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
                'HighChol', 'CholCheck', 'PhysActivity', 'Fruits',
                'Veggies', 'HvyAlcoholConsump', 'Smoker', 'Sex',
                'DiffWalk', 'HighBP']
X = df[feature_cols]
y = df['Diabetes_binary']

# SMOTE needed: 86/14 imbalance found on Day 3
X_train, X_test, y_train, y_test = split_and_smote(X, y, 'Diabetes', use_smote=True)
save_splits(X_train, X_test, y_train, y_test, 'diabetes')
print("Features used:", feature_cols)


# ============================================================
# 2. HYPERTENSION — Logistic Regression (needs scaling)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING: HYPERTENSION")
print("="*60)
# Same BRFSS source, different target
df = pd.read_csv('data/diabetes/cleaned_diabetes_brfss.csv')
feature_cols = ['Age', 'BMI', 'GenHlth', 'HighChol', 'Smoker',
                'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk',
                'MentHlth', 'PhysHlth', 'Sex']
X = df[feature_cols]
y = df['HighBP']

# Classes are 57/43 — balanced enough, SMOTE optional but applied for consistency
X_train, X_test, y_train, y_test = split_and_smote(X, y, 'Hypertension', use_smote=False)
X_train_sc, X_test_sc = scale_features(X_train, X_test, 'hypertension')
save_splits(X_train_sc, X_test_sc, y_train, y_test, 'hypertension')
print("Features used:", feature_cols)


# ============================================================
# 3. HEART DISEASE — Random Forest (tree-based, no scaling)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING: HEART DISEASE")
print("="*60)
df = pd.read_csv('data/heart/cleaned_heart_restricted.csv')
feature_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs']
X = df[feature_cols]
y = df['heart_disease_binary']

# Classes are 54/46 — well balanced, SMOTE not needed
X_train, X_test, y_train, y_test = split_and_smote(X, y, 'Heart Disease', use_smote=False)
save_splits(X_train, X_test, y_train, y_test, 'heart')
print("Features used:", feature_cols)


# ============================================================
# 4. OBESITY — Gradient Boosting (tree-based, no scaling)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING: OBESITY")
print("="*60)
df = pd.read_csv('data/obesity/cleaned_obesity.csv')

# Encode categoricals
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
yes_no_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in yes_no_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# CAEC and CALC are ordinal — encode with meaningful order
caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(caec_map)
df['CALC'] = df['CALC'].map(calc_map)

# MTRANS — one-hot encode
df = pd.get_dummies(df, columns=['MTRANS'])
mtrans_cols = [c for c in df.columns if c.startswith('MTRANS_')]

feature_cols = ['Gender', 'Age', 
                'family_history_with_overweight', 'FAVC', 'FCVC',
                'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                'TUE', 'CALC'] + mtrans_cols
X = df[feature_cols]
y = df['obesity_risk']

# 73/27 imbalance — apply SMOTE
X_train, X_test, y_train, y_test = split_and_smote(X, y, 'Obesity', use_smote=True)
save_splits(X_train, X_test, y_train, y_test, 'obesity')
print("Features used:", feature_cols)

# Save the feature column order so backend can reconstruct correctly
joblib.dump(feature_cols, 'models/obesity_feature_cols.pkl')


# ============================================================
# 5. STRESS — Decision Tree (tree-based, no scaling)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING: STRESS")
print("="*60)
df = pd.read_csv('data/stress/cleaned_stress.csv')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

feature_cols = ['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'systolic_bp']
X = df[feature_cols]
y = df['high_stress']

# 62/38 — reasonably balanced, no SMOTE needed
X_train, X_test, y_train, y_test = split_and_smote(X, y, 'Stress', use_smote=False)
save_splits(X_train, X_test, y_train, y_test, 'stress')
print("Features used:", feature_cols)


print("\n=== DAY 5 PREPROCESSING COMPLETE ===")
print("Splits saved in data/processed/")
print("Scalers (where applicable) saved in models/")
