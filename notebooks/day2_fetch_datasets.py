import pandas as pd
from ucimlrepo import fetch_ucirepo

# Diabetes + Hypertension (same source dataset, two different targets)
diabetes = fetch_ucirepo(id=891)
diabetes_df = pd.concat([diabetes.data.features, diabetes.data.targets], axis=1)
diabetes_df.to_csv('data/diabetes/diabetes_brfss.csv', index=False)
print("Diabetes/Hypertension dataset shape:", diabetes_df.shape)
print("Diabetes/Hypertension columns:", diabetes_df.columns.tolist())
print()

# Heart Disease
heart = fetch_ucirepo(id=45)
heart_df = pd.concat([heart.data.features, heart.data.targets], axis=1)
heart_df.to_csv('data/heart/heart_uci.csv', index=False)
print("Heart Disease dataset shape:", heart_df.shape)
print("Heart Disease columns:", heart_df.columns.tolist())
print()

# Obesity
obesity = fetch_ucirepo(id=544)
obesity_df = pd.concat([obesity.data.features, obesity.data.targets], axis=1)
obesity_df.to_csv('data/obesity/obesity_uci.csv', index=False)
print("Obesity dataset shape:", obesity_df.shape)
print("Obesity columns:", obesity_df.columns.tolist())
print()

print("Diabetes/Hypertension sample:")
print(diabetes_df.head())
print()
print("Heart Disease sample:")
print(heart_df.head())
print()
print("Obesity sample:")
print(obesity_df.head())
