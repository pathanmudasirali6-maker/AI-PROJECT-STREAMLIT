# AI Teachable Machine Clone

A full stack teachable machine clone built with Streamlit on the frontend and FastAPI on the backend.

## Project Structure

```text
project/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── predict_service.py
│   │   └── train_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── model_utils.py
│   ├── dataset/
│   └── saved_models/
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

## Backend Endpoints

- `POST /upload-sample` - upload training images for a class
- `POST /train` - train the model on uploaded samples
- `POST /predict` - predict a class for a test image
- `GET /dataset-stats` - get dataset metrics
- `GET /health` - health check endpoint

## Setup in VS Code

### 1. Create a Python virtual environment

```powershell
cd c:\Users\Ahsan-Home\Desktop\PROJECT\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd ..\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Run backend

```powershell
cd c:\Users\Ahsan-Home\Desktop\PROJECT
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run frontend

```powershell
cd c:\Users\Ahsan-Home\Desktop\PROJECT\frontend
streamlit run app.py
```

## Docker Setup

Build and run both services with Docker Compose:

```powershell
docker compose up --build
```

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

## Notes

- Uploaded training samples are stored in `backend/dataset`.
- Trained model artifacts are persisted to `backend/saved_models`.
- The frontend communicates with the backend through HTTP requests.
- The dashboard uses a dark theme, status messages, and prediction probability charts.
