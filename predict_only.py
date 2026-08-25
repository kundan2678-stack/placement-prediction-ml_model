"""
================================================================================
PREDICT ONLY - Bina training ke sirf prediction
Pehle 'placement_pipeline.py' run karo (model save hoga), phir yeh file chalao
================================================================================
"""

import joblib
import pandas as pd
import numpy as np

# Load saved model and helpers
print("Loading saved model...")
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')
le_target = joblib.load('target_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')
print("Model loaded successfully!\n")

def predict_placement(iq, prev_sem, cgpa, academic_perf, internship, 
                       extra_curricular, communication, projects):
    """
    Kisi bhi student ka placement predict karo.

    Example:
        result = predict_placement(120, 8.5, 8.8, 8, 'Yes', 7, 8, 3)
        print(result)
    """
    # Encode internship
    intern_enc = 1 if str(internship).lower() == 'yes' else 0

    # Build feature row
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

    # Same column order as training
    input_data = input_data[feature_names]

    # Predict
    proba = model.predict_proba(input_data)[0]
    pred = model.predict(input_data)[0]
    label = le_target.inverse_transform([pred])[0]

    placed_prob = proba[1] * 100

    # Visual output
    bar = "█" * int(placed_prob / 5) + "░" * (20 - int(placed_prob / 5))

    print("=" * 50)
    print("         PLACEMENT PREDICTION RESULT")
    print("=" * 50)
    print(f"  Placement Status : {'✅ PLACED' if label == 'Yes' else '❌ NOT PLACED'}")
    print(f"  Confidence       : {max(proba)*100:.1f}%")
    print(f"  Placement Chance : {placed_prob:.1f}% [{bar}]")
    print("=" * 50)

    return {
        'Placement': 'Placed' if label == 'Yes' else 'Not Placed',
        'Confidence': f"{max(proba)*100:.1f}%",
        'Chance_of_Placement': f"{placed_prob:.1f}%"
    }


# ========================= DEMO =========================
if __name__ == "__main__":
    print("\n🔮 Student 1: High Achiever")
    predict_placement(125, 9.2, 9.5, 9, 'Yes', 8, 9, 4)

    print("\n🔮 Student 2: Average")
    predict_placement(95, 7.0, 7.2, 6, 'No', 5, 6, 2)

    print("\n🔮 Student 3: Needs Improvement")
    predict_placement(70, 5.5, 5.3, 3, 'No', 2, 3, 0)

    print("\n💡 Apna student check karo:")
    print("   predict_placement(iq, prev_sem, cgpa, academic, internship, extra, comm, projects)")
