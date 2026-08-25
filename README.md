# 🎓 Student Placement Prediction ML

## 🎯 Project Objective

The main objective is to build an ML-based system that can provide an early estimate of a student's placement outcome using measurable academic, skill, and experience-related attributes.

A machine learning project that predicts whether a college student is likely to get placed based on academic performance, IQ, internship experience, communication skills, extracurricular activities, and completed projects.

The project includes a complete ML pipeline for **data preprocessing, feature engineering, model training, model comparison, hyperparameter tuning, model persistence, and student-level prediction**.

---

## 🚀 Project Overview

Student placement depends on multiple academic and skill-related factors. This project uses machine learning classification algorithms to analyze student information and predict:

* ✅ Whether the student is likely to be placed
* 📊 Placement probability
* 🎯 Prediction confidence

The system compares multiple machine learning models and uses **Gradient Boosting with hyperparameter tuning** as the final prediction model.

---

## ✨ Key Features

* 📂 Load and process student placement dataset
* 🧹 Data preprocessing
* 📉 IQ outlier handling using percentile capping
* 🔢 Categorical encoding for internship experience and placement
* 🧠 Feature engineering
* 📊 Train-test split with stratification
* ⚖️ Class imbalance handling
* 🤖 Multiple ML model comparison
* 🔍 Hyperparameter tuning using GridSearchCV
* 💾 Save trained ML model using Joblib
* 🔮 Predict placement for individual students
* 📈 Return placement probability and prediction confidence

---

## 🧠 Machine Learning Workflow

```text
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
```

---

## 📊 Features Used

The prediction system uses the following student attributes:

| Feature                  | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `IQ`                     | Student IQ score                                    |
| `Prev_Sem_Result`        | Previous semester result                            |
| `CGPA`                   | Current CGPA                                        |
| `Academic_Performance`   | Academic performance score                          |
| `Internship_Experience`  | Whether the student has internship experience       |
| `Extra_Curricular_Score` | Extracurricular activity score                      |
| `Communication_Skills`   | Communication skills score                          |
| `Projects_Completed`     | Number of completed projects                        |
| `Academic_Average`       | Average of CGPA and previous semester result        |
| `Soft_Skills_Score`      | Average of communication and extracurricular scores |

### Feature Engineering

Two additional features are engineered during preprocessing:

```text
Academic_Average = (CGPA + Prev_Sem_Result) / 2

Soft_Skills_Score = (Communication_Skills + Extra_Curricular_Score) / 2
```

These transformations are implemented in the training pipeline.

---

## 🤖 Machine Learning Models

The project evaluates three classification algorithms:

### 1. Logistic Regression

Used as a baseline classification model with class balancing.

### 2. Random Forest

An ensemble tree-based classifier configured with class balancing.

### 3. Gradient Boosting

A boosting-based classifier trained with sample weights to address class imbalance.

### Model Evaluation Metrics

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-Score

The implementation calculates these metrics for all three models.

---

## 🔍 Hyperparameter Tuning

Gradient Boosting is further optimized using **GridSearchCV**.

The search explores:

```text
n_estimators:
    200, 300

max_depth:
    5, 7

learning_rate:
    0.1, 0.15
```

The tuning process uses **3-fold cross-validation** and **F1-score** as the optimization metric.

---

## ⚖️ Handling Class Imbalance

The project calculates the imbalance ratio from the training data and uses weighting techniques during model training.

* Gradient Boosting receives sample weights based on the class distribution.
* Logistic Regression uses balanced class weights.
* Random Forest uses balanced class weights.

---

## 💾 Saved Model Files

After training, the following files are generated:

```text
placement_model.pkl
scaler.pkl
target_encoder.pkl
internship_encoder.pkl
feature_names.pkl
```

These files store the trained model and supporting preprocessing information required for prediction.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

### Machine Learning

* Logistic Regression
* Random Forest
* Gradient Boosting
* GridSearchCV
* StandardScaler
* LabelEncoder

---

## 📁 Project Structure

```text
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
```

> The `.pkl` files are generated after running the training pipeline.

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/kundan2678-stack/placement-prediction-ml_model.git
```

### Move into the Project Directory

```bash
cd placement-prediction-ml_model
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

The required Python libraries are also listed in the training pipeline.

---

## 🚀 Train the Model

Run:

```bash
python placement_pipeline.py
```

This will:

1. Load the dataset
2. Preprocess the data
3. Engineer additional features
4. Split the dataset
5. Train multiple ML models
6. Compare model performance
7. Tune Gradient Boosting
8. Evaluate the final model
9. Save the trained model and preprocessing files

---

## 🔮 Make Predictions

After training the model, run:

```bash
python predict_only.py
```

The prediction script loads the previously saved model and supporting files without retraining the model.

### Example

```python
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
```

### Input Parameters

The function accepts:

* IQ
* Previous Semester Result
* CGPA
* Academic Performance
* Internship Experience
* Extra-Curricular Score
* Communication Skills
* Projects Completed

### Prediction Output

The prediction output includes:

* Placement
* Confidence
* Chance_of_Placement

---

## 📌 Example Prediction

```text
==================================================
         PLACEMENT PREDICTION RESULT
==================================================
  Placement Status : ✅ PLACED
  Confidence       : XX.X%
  Placement Chance : XX.X%
==================================================
```

> The actual values depend on the trained model and input student data.

---

## 📈 Model Evaluation

The final model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

The project prioritizes **F1-Score** during hyperparameter tuning because the training pipeline explicitly uses F1 scoring in `GridSearchCV`.

To see the actual performance values for your current dataset, run:

```bash
python placement_pipeline.py
```

---

## 🎯 Future Improvements

* 🌐 Build a web application for real-time prediction
* 📊 Interactive student analytics dashboard
* 📈 Feature importance visualization
* 🧠 Explainable AI using SHAP
* 📋 Personalized improvement recommendations
* ☁️ Deploy the model as a REST API
* 🚀 Deploy the complete application online
* 🔄 Automated model retraining with new placement data

---

## ⚠️ Disclaimer

This project is intended for **educational and analytical purposes**.

The prediction represents a machine learning estimate based on the available dataset and should not be considered a guaranteed placement outcome. Real-world placement decisions depend on many additional factors that may not be represented in the dataset.

---

## 👨‍💻 Author

**Kundan**

GitHub:
https://github.com/kundan2678-stack

Project Repository:
https://github.com/kundan2678-stack/placement-prediction-ml_model

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

**Built with Python & Machine Learning 🚀**
