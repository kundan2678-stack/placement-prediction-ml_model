# 🎓 Student Placement Prediction ML
🎯 Project Objective

The main objective is to build an ML-based system that can provide an early estimate of a student's placement outcome using measurable academic, skill, and experience-related attributes.

A machine learning project that predicts whether a college student is likely to get placed based on academic performance, IQ, internship experience, communication skills, extracurricular activities, and completed projects.

The project includes a complete ML pipeline for data preprocessing, feature engineering, model training, model comparison, hyperparameter tuning, model persistence, and student-level prediction.

🚀 Project Overview

Student placement depends on multiple academic and skill-related factors. This project uses machine learning classification algorithms to analyze student information and predict:

✅ Whether the student is likely to be placed
📊 Placement probability
🎯 Prediction confidence

The system compares multiple machine learning models and uses Gradient Boosting with hyperparameter tuning as the final prediction model.

✨ Key Features
📂 Load and process student placement dataset
🧹 Data preprocessing
📉 IQ outlier handling using percentile capping
🔢 Categorical encoding for internship experience and placement
🧠 Feature engineering
📊 Train-test split with stratification
⚖️ Class imbalance handling
🤖 Multiple ML model comparison
🔍 Hyperparameter tuning using GridSearchCV
💾 Save trained ML model using Joblib
🔮 Predict placement for individual students
📈 Return placement probability and prediction confidence
🧠 Machine Learning Workflow
Student Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Preprocessing
       │
       ├── Remove College_ID
       ├── Handle IQ Outliers
       └── Encode Categorical Values
       │
       ▼
Feature Engineering
       │
       ├── Academic_Average
       └── Soft_Skills_Score
       │
       ▼
Train-Test Split
       │
       ▼
Model Training
       │
       ├── Logistic Regression
       ├── Random Forest
       └── Gradient Boosting
       │
       ▼
Model Comparison
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Best Gradient Boosting Model
       │
       ▼
Model Saving
       │
       ▼
Student Placement Prediction
📊 Features Used

The prediction system uses the following student attributes:

Feature	Description
IQ	Student IQ score
Prev_Sem_Result	Previous semester result
CGPA	Current CGPA
Academic_Performance	Academic performance score
Internship_Experience	Whether the student has internship experience
Extra_Curricular_Score	Extracurricular activity score
Communication_Skills	Communication skills score
Projects_Completed	Number of completed projects
Academic_Average	Average of CGPA and previous semester result
Soft_Skills_Score	Average of communication and extracurricular scores

Two additional features are engineered during preprocessing:

Academic_Average = (CGPA + Prev_Sem_Result) / 2

Soft_Skills_Score = (Communication_Skills + Extra_Curricular_Score) / 2

These transformations are implemented in the training pipeline.

🤖 Machine Learning Models

The project evaluates three classification algorithms:

1. Logistic Regression

Used as a baseline classification model with class balancing.

2. Random Forest

An ensemble tree-based classifier configured with class balancing.

3. Gradient Boosting

A boosting-based classifier trained with sample weights to address class imbalance.

The models are compared using:

Accuracy
Precision
Recall
F1-Score

The implementation calculates these metrics for all three models.

🔍 Hyperparameter Tuning

Gradient Boosting is further optimized using GridSearchCV.

The search explores:

n_estimators:
    200, 300

max_depth:
    5, 7

learning_rate:
    0.1, 0.15

The tuning process uses 3-fold cross-validation and F1-score as the optimization metric.

⚖️ Handling Class Imbalance

The project calculates the imbalance ratio from the training data and uses weighting techniques during model training.

Gradient Boosting receives sample weights based on the class distribution, while Logistic Regression and Random Forest use balanced class weights.

💾 Saved Model Files

After training, the following files are generated:

placement_model.pkl
scaler.pkl
target_encoder.pkl
internship_encoder.pkl
feature_names.pkl

These files store the trained model and supporting preprocessing information required for prediction.

🛠️ Tech Stack
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Machine Learning
Logistic Regression
Random Forest
Gradient Boosting
GridSearchCV
StandardScaler
LabelEncoder
📁 Project Structure
placement-prediction-ml_model/
│
├── college_student_placement_dataset.csv
├── placement_pipeline.py
├── predict_only.py
├── src.ipynb
│
├── placement_model.pkl
├── scaler.pkl
├── target_encoder.pkl
├── internship_encoder.pkl
├── feature_names.pkl
│
└── README.md

The .pkl files are generated after running the training pipeline.

⚙️ Installation

Clone the repository:

git clone https://github.com/kundan2678-stack/placement-prediction-ml_model.git

Move into the project directory:

cd placement-prediction-ml_model

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

The required Python libraries are also listed in the training pipeline.

🚀 Train the Model

Run:

python placement_pipeline.py

This will:

Load the dataset
Preprocess the data
Engineer additional features
Split the dataset
Train multiple ML models
Compare model performance
Tune Gradient Boosting
Evaluate the final model
Save the trained model and preprocessing files
🔮 Make Predictions

After training the model, run:

python predict_only.py

The prediction script loads the previously saved model and supporting files without retraining the model.

Example:

predict_placement(
    120,
    8.5,
    8.8,
    8,
    "Yes",
    7,
    8,
    3
)

The function accepts:

IQ
Previous Semester Result
CGPA
Academic Performance
Internship Experience
Extra-Curricular Score
Communication Skills
Projects Completed

The prediction output includes:

Placement
Confidence
Chance_of_Placement




📌 Example Prediction
==================================================
         PLACEMENT PREDICTION RESULT
==================================================
  Placement Status : ✅ PLACED
  Confidence       : XX.X%
  Placement Chance : XX.X%
==================================================

The actual values depend on the trained model and input student data.

📈 Model Evaluation

The final model is evaluated using:

Accuracy
Precision
Recall
F1-Score

The project prioritizes F1-Score during hyperparameter tuning because the training pipeline explicitly uses F1 scoring in GridSearchCV.

To see the actual performance values for your current dataset, run:

python placement_pipeline.py
