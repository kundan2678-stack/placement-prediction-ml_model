# PlacementIQ API

This document summarizes the backend API contract, request/response payloads, and model behavior.

## Base URL
- Local: `http://localhost:8000`
- Production: configured in environment

## Endpoints

### GET /api/health
Returns application and model status.

Example response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "message": "PlacementIQ API is running."
}
```

### GET /api/model-info
Returns metadata about the trained model, supported features, and deployment state.

Example response:
```json
{
  "model_available": true,
  "model_name": "Gradient Boosting Classifier",
  "description": "Original placement prediction model preserved from the repository pipeline.",
  "feature_names": [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
    "Academic_Average",
    "Soft_Skills_Score"
  ],
  "supports_probability": true,
  "positive_class_label": "Yes",
  "algorithms": [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting"
  ]
}
```

### POST /api/predict
Submits a single prediction request.

Request body:
```json
{
  "iq": 116,
  "previous_semester_result": 8.4,
  "cgpa": 8.7,
  "academic_performance": 8,
  "internship": true,
  "extracurricular_score": 7,
  "communication_skills": 8,
  "projects_completed": 3
}
```

Example successful response:
```json
{
  "success": true,
  "prediction": "Placed",
  "probability": 0.78,
  "confidence": 0.78,
  "model_name": "Gradient Boosting Classifier",
  "message": "Prediction generated using the original repository model.",
  "input_summary": {
    "iq": 116,
    "cgpa": 8.7,
    "internship": true,
    "projects_completed": 3
  }
}
```

## Validation rules
- `iq` must be between 40 and 160.
- `previous_semester_result` must be between 4 and 10.
- `cgpa` must be between 4 and 10.5.
- `academic_performance` must be between 1 and 10.
- `extracurricular_score` must be between 0 and 10.
- `communication_skills` must be between 1 and 10.
- `projects_completed` must be between 0 and 5.

## Probability semantics
The backend uses the original model's `predict_proba` output. When the target encoder identifies the positive class as `Yes`, the probability returned is the model's probability for the positive placement class, not a calibrated guarantee.

## Error handling
The API returns 400-level validation errors for malformed or out-of-range values and 500-level errors only for internal inference failures. Sensitive details are intentionally suppressed.
