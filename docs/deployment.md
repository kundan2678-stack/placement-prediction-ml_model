# PlacementIQ architecture

## System overview
PlacementIQ follows a simple full-stack architecture:

- Frontend: React + Vite + TypeScript + Tailwind
- Backend: FastAPI + Python
- Model layer: original repository model and preprocessing artifacts
- Data flow: browser form -> API -> model service -> prediction -> result UI

## Frontend
The frontend is a multipage SaaS-style app with:
- Landing page
- Dashboard overview
- Prediction form
- Results page
- Improvement plan
- About model

The app uses local browser storage for recent prediction history and simple user preferences.

## Backend
The backend provides:
- application configuration
- health checks
- model metadata
- prediction endpoint
- recommendation service
- CORS and validation middleware

## Model integration
The original training script saves `placement_model.pkl`, `target_encoder.pkl`, and `feature_names.pkl`. The backend app loads these once at startup and uses them for inference. It does not retrain the model on each request.

## Data flow
1. Student enters academic details.
2. Frontend sends payload to `/api/predict`.
3. FastAPI validates the input.
4. Model service builds feature rows in the exact training order.
5. Model predicts placement and probability.
6. Recommendation engine analyzes the same input data.
7. Response is returned and rendered by the frontend.

## Deployment landscape
- Frontend deployed on Vercel
- Backend deployed on Render
- API URL stored in frontend environment variables
- CORS configured through environment values
