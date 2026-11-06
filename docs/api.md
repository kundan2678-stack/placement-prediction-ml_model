# PlacementIQ deployment guide

## Frontend deployment (Vercel)
1. Import the repository into Vercel.
2. Set the project root to `frontend`.
3. Configure build command:
```bash
npm run build
```
4. Set output directory:
```bash
dist
```
5. Add environment variable:
```env
VITE_API_BASE_URL=https://your-render-app-url.onrender.com
```

## Backend deployment (Render)
1. Create a new Web Service in Render.
2. Connect the repository.
3. Set root directory to `backend`.
4. Use a Python runtime.
5. Add the following start command:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```
6. Add environment variables:
```env
APP_ENV=production
CORS_ORIGINS=https://your-frontend.vercel.app
MODEL_ROOT_PATH=..
```

## Model artifact handling
The deployed backend must have access to the root project directory and the generated `.pkl` files. If model files are missing, run the original training script from the repo root before starting the API service.

## Health checks
Use Render health check endpoint:
```text
/api/health
```
