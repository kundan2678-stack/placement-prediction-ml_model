"""
================================================================================
STUDENT PLACEMENT PREDICTION - COMPLETE PIPELINE
One file mein sab kuch: EDA + Preprocessing + Training + Prediction
================================================================================
"""

# ========================= STEP 0: INSTALL (Terminal mein run karo) =========================
# pip install pandas numpy matplotlib seaborn scikit-learn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# ========================= STEP 1: LOAD DATA =========================
# Apne CSV file ka path yahan daalo
FILE_PATH = 'college_student_placement_dataset.csv'  # <-- YEH PATH CHANGE KARO

df = pd.read_csv(FILE_PATH)
print(f"Data loaded: {df.shape[0]} students, {df.shape[1]} columns")

# ========================= STEP 2: PREPROCESSING =========================
df_model = df.copy()

# Drop ID column
df_model = df_model.drop(columns=['College_ID'])

# Fix IQ outliers (cap at 1st & 99th percentile)
iq_low, iq_high = df_model['IQ'].quantile([0.01, 0.99])
df_model['IQ'] = df_model['IQ'].clip(iq_low, iq_high)

# Encode Yes/No to 1/0
le_internship = LabelEncoder()
df_model['Internship_Experience'] = le_internship.fit_transform(df_model['Internship_Experience'])

le_target = LabelEncoder()
df_model['Placement'] = le_target.fit_transform(df_model['Placement'])

# Feature Engineering
df_model['Academic_Average'] = (df_model['CGPA'] + df_model['Prev_Sem_Result']) / 2
df_model['Soft_Skills_Score'] = (df_model['Communication_Skills'] + df_model['Extra_Curricular_Score']) / 2

# Features & Target
X = df_model.drop(columns=['Placement'])
y = df_model['Placement']
feature_names = list(X.columns)

print(f"Features: {feature_names}")
print(f"Target distribution: {dict(y.value_counts())}")

# ========================= STEP 3: TRAIN-TEST SPLIT =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imbalance weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

# ========================= STEP 4: TRAIN MODELS =========================
print("\nTraining models...")

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Model 2: Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Model 3: Gradient Boosting (with sample weights for imbalance)
sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    min_samples_split=5, subsample=0.9, random_state=42
)
gb.fit(X_train, y_train, sample_weight=sample_weights)
gb_pred = gb.predict(X_test)

# ========================= STEP 5: COMPARE & SELECT BEST =========================
def get_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy': f"{accuracy_score(y_true, y_pred):.2%}",
        'Precision': f"{precision_score(y_true, y_pred):.2%}",
        'Recall': f"{recall_score(y_true, y_pred):.2%}",
        'F1': f"{f1_score(y_true, y_pred):.2%}"
    }

results = [
    get_metrics('Logistic Regression', y_test, lr_pred),
    get_metrics('Random Forest', y_test, rf_pred),
    get_metrics('Gradient Boosting', y_test, gb_pred)
]

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for r in results:
    print(f"{r['Model']:20s} | Acc: {r['Accuracy']} | F1: {r['F1']}")

# Pick best by F1
best_model = gb  # Usually GB wins; change if needed
best_name = 'Gradient Boosting'

# ========================= STEP 6: HYPERPARAMETER TUNING (Optional) =========================
print("\nTuning best model...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.15]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42, min_samples_split=5, subsample=0.9),
    param_grid, cv=3, scoring='f1', n_jobs=-1
)
grid.fit(X_train, y_train, sample_weight=sample_weights)

best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# Final evaluation
final_pred = best_model.predict(X_test)
print("\nFinal Model Performance:")
print(f"Accuracy : {accuracy_score(y_test, final_pred):.4f}")
print(f"Precision: {precision_score(y_test, final_pred):.4f}")
print(f"Recall   : {recall_score(y_test, final_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, final_pred):.4f}")

# ========================= STEP 7: SAVE EVERYTHING =========================
joblib.dump(best_model, 'placement_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_target, 'target_encoder.pkl')
joblib.dump(le_internship, 'internship_encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("\nSaved files:")
print("  - placement_model.pkl (trained model)")
print("  - scaler.pkl (feature scaler)")
print("  - target_encoder.pkl (Yes/No encoder)")
print("  - internship_encoder.pkl (Internship encoder)")
print("  - feature_names.pkl (feature order)")

# ========================= STEP 8: PREDICT FUNCTION =========================
def predict_placement(iq, prev_sem, cgpa, academic_perf, internship, 
                       extra_curricular, communication, projects):
    """
    Kisi bhi student ka placement predict karo.

    Parameters:
        iq: int (50-160)
        prev_sem: float (4-10)
        cgpa: float (4-10)
        academic_perf: int (1-10)
        internship: str ('Yes' ya 'No')
        extra_curricular: int (0-10)
        communication: int (0-10)
        projects: int (0-5)

    Returns:
        dict with prediction and probabilities
    """
    # Encode internship
    intern_enc = 1 if internship.lower() == 'yes' else 0

    # Build features in EXACT same order as training
    input_data = pd.DataFrame([{
        'IQ': iq,
        'Prev_Sem_Result': prev_sem,
        'CGPA': cgpa,
        'Academic_Performance': academic_perf,
        'Internship_Experience': intern_enc,
        'Extra_Curricular_Score': extra_curricular,
        'Communication_Skills': communication,
        'Projects_Completed': projects,
        'Academic_Average': (cgpa + prev_sem) / 2,
        'Soft_Skills_Score': (communication + extra_curricular) / 2
    }])

    # Ensure column order
    input_data = input_data[feature_names]

    # Predict
    proba = best_model.predict_proba(input_data)[0]
    pred = best_model.predict(input_data)[0]

    label = le_target.inverse_transform([pred])[0]

    return {
        'Placement': '✅ PLACED' if label == 'Yes' else '❌ NOT PLACED',
        'Confidence': f"{max(proba)*100:.1f}%",
        'Chance_of_Placement': f"{proba[1]*100:.1f}%"
    }

# ========================= DEMO =========================
print("\n" + "="*60)
print("DEMO PREDICTION")
print("="*60)

# Example 1: Strong student
student1 = predict_placement(
    iq=125, prev_sem=9.2, cgpa=9.5, academic_perf=9,
    internship='Yes', extra_curricular=8, communication=9, projects=4
)
print(f"\nStudent 1 (Strong): {student1}")

# Example 2: Average student
student2 = predict_placement(
    iq=95, prev_sem=7.0, cgpa=7.2, academic_perf=6,
    internship='No', extra_curricular=5, communication=6, projects=2
)
print(f"Student 2 (Average): {student2}")

# Example 3: Weak student
student3 = predict_placement(
    iq=70, prev_sem=5.5, cgpa=5.3, academic_perf=3,
    internship='No', extra_curricular=2, communication=3, projects=0
)
print(f"Student 3 (Weak): {student3}")

print("\n" + "="*60)
print("READY! Ab tum predict_placement() function se kisi bhi student ka placement check kar sakte ho.")
print("="*60)
