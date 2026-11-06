# PlacementIQ

An AI-powered student placement prediction platform built on the original machine learning model in this repository.

## Overview
PlacementIQ turns the existing placement prediction project into a full-stack MVP web application. Students can enter academic and skill details, run a prediction against the original trained model, read an explanation of the result, and review actionable improvement recommendations.

## Goals
- Preserve the original ML workflow and dataset.
- Expose the model through a FastAPI backend.
- Provide a polished React + Tailwind frontend.
- Allow students to understand predictions and improvement opportunities.
- Keep the project deployable and easy to run locally.

## Product Value
The platform helps students estimate their placement readiness before the placement season begins. It highlights both strengths and weak areas, and it translates model output into understandable, practical actions.

## Features
- FastAPI backend for health checks and predictions.
- Original model integration with feature-order preservation.
- Student placement form with validation.
- Probability-aware results display.
- Personalized recommendation engine.
- Responsive dashboard and landing page.
- Session-local prediction history.

## Original ML model preserved
This project keeps the existing ML scripts and data intact:
- `placement_pipeline.py`
- `predict_only.py`
- `src.ipynb`
- `college_student_placement_dataset.csv`

The backend loads the trained artifacts produced by the existing training pipeline instead of replacing the model.

## Repository structure
```text
placement-prediction-ml_model/
├── README.md
├── .env.example
├── .gitignore
├── college_student_placement_dataset.csv
├── placement_pipeline.py
├── predict_only.py
├── src.ipynb
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── public/
│   ├── src/
│   ├── .env.example
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
└── LICENSE (if added later)
```

## Tech stack
### Backend
- Python
- FastAPI
- Pydantic
- Pandas
- NumPy
- scikit-learn
- Joblib

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Axios
- Recharts
- Lucide React

## Model details
The repository original pipeline uses:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Feature engineering for `Academic_Average` and `Soft_Skills_Score`
- `predict_proba` for probability output

The exact model metric values are generated when the original training script is run on the current dataset. This project does not invent or hardcode performance values.

## Local setup

### 1) Clone repo
```bash
git clone https://github.com/kundan2678-stack/placement-prediction-ml_model.git
cd placement-prediction-ml_model
```

### 2) Backend environment
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Generate model artifacts (if missing)
```bash
cd ..
python placement_pipeline.py
```
This uses the original training workflow and persists the required `.pkl` artifacts in the repository root.

### 4) Run backend
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) Frontend environment
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## Environment variables
### Backend
Copy `.env.example` and update values as needed.

### Frontend
```env
VITE_API_BASE_URL=http://localhost:8000
```

## API endpoints
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`

## Testing
```bash
cd backend
pytest
```

Frontend build validation:
```bash
cd frontend
npm run build
```

## Deployment notes
This repo is prepared for a Vercel frontend and Render backend deployment. Configure environment variables for the production API origin in the deployed frontend and backend CORS origins in the deployed backend.

## Screenshots
Add screenshots here after running the app locally.

## Limitations
- This MVP does not include authentication.
- The prediction is statistical and should not be treated as a guarantee.
- The model remains dependent on the original training dataset and preprocessing logic.

## Disclaimer
The placement prediction is an educational and analytical estimate based on the repository’s existing data and machine learning pipeline. It should not be considered a guaranteed placement decision.
