import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

# 1. Imbalanced Credit Card Default / Fraud Dataset (10% minority)
X, y = make_classification(
    n_samples=1200,
    n_features=8,
    n_informative=6,
    n_redundant=1,
    weights=[0.88, 0.12],  # ~12% minority
    random_state=42,
    flip_y=0.01
)
feature_cols = ['Transaction_Amount', 'Age', 'Credit_Score', 'Account_Balance', 
                'Monthly_Expenses', 'Debt_Ratio', 'Num_Accounts', 'Risk_Score']
df_credit = pd.DataFrame(X, columns=feature_cols)
# make some features look realistic
df_credit['Transaction_Amount'] = np.round(np.abs(df_credit['Transaction_Amount'] * 250 + 150), 2)
df_credit['Age'] = np.clip(np.round(df_credit['Age'] * 10 + 40), 18, 75).astype(int)
df_credit['Credit_Score'] = np.clip(np.round(df_credit['Credit_Score'] * 60 + 680), 300, 850).astype(int)
df_credit['Account_Balance'] = np.round(np.abs(df_credit['Account_Balance'] * 5000 + 4500), 2)
df_credit['Monthly_Expenses'] = np.round(np.abs(df_credit['Monthly_Expenses'] * 800 + 1200), 2)
df_credit['Debt_Ratio'] = np.clip(np.round(np.abs(df_credit['Debt_Ratio'] * 0.2 + 0.35), 3), 0.01, 1.0)
df_credit['Num_Accounts'] = np.clip(np.round(np.abs(df_credit['Num_Accounts'] * 2 + 3)), 1, 12).astype(int)
df_credit['Risk_Score'] = np.round(np.abs(df_credit['Risk_Score'] * 15 + 50), 2)
df_credit['Default_Status'] = y  # 0 = No Default, 1 = Defaulted
df_credit.to_csv('sample_data/credit_default_imbalanced.csv', index=False)
print("Saved credit_default_imbalanced.csv, shape:", df_credit.shape, "Class counts:\n", df_credit['Default_Status'].value_counts())

# 2. Medical Healthcare Diagnosis (Imbalanced 15% minority)
X_med, y_med = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=5,
    n_redundant=1,
    weights=[0.82, 0.18],  # ~18% minority
    random_state=101
)
med_cols = ['Glucose_Level', 'Blood_Pressure', 'BMI', 'Insulin', 'Cholesterol', 'Heart_Rate']
df_med = pd.DataFrame(X_med, columns=med_cols)
df_med['Glucose_Level'] = np.round(np.abs(df_med['Glucose_Level'] * 30 + 115), 1)
df_med['Blood_Pressure'] = np.round(np.abs(df_med['Blood_Pressure'] * 15 + 78), 1)
df_med['BMI'] = np.round(np.abs(df_med['BMI'] * 5 + 26.5), 1)
df_med['Insulin'] = np.round(np.abs(df_med['Insulin'] * 40 + 95), 1)
df_med['Cholesterol'] = np.round(np.abs(df_med['Cholesterol'] * 35 + 190), 1)
df_med['Heart_Rate'] = np.round(np.abs(df_med['Heart_Rate'] * 12 + 72), 0).astype(int)
df_med['Disease_Outcome'] = y_med
df_med.to_csv('sample_data/medical_diagnosis_imbalanced.csv', index=False)
print("Saved medical_diagnosis_imbalanced.csv, shape:", df_med.shape, "Class counts:\n", df_med['Disease_Outcome'].value_counts())

# 3. Balanced Dataset (50-50 class distribution) to test option 1 & 2
X_bal, y_bal = make_classification(
    n_samples=800,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    weights=[0.50, 0.50],  # 50-50 Balanced!
    random_state=7
)
bal_cols = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E']
df_bal = pd.DataFrame(X_bal, columns=bal_cols)
for c in bal_cols:
    df_bal[c] = np.round(df_bal[c], 3)
df_bal['Target_Class'] = y_bal
df_bal.to_csv('sample_data/balanced_dataset_sample.csv', index=False)
print("Saved balanced_dataset_sample.csv, shape:", df_bal.shape, "Class counts:\n", df_bal['Target_Class'].value_counts())
