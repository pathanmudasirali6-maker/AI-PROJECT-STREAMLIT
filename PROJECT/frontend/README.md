# AI Teachable Machine Clone

A full stack teachable machine clone with a Streamlit frontend and FastAPI backend. The application allows users to upload image samples per class, train a MobileNetV3 feature extractor with a scikit-learn classifier, and perform predictions.

## Project Structure

```text
project/
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── services/
│   │   ├── train_service.py
│   │   └── predict_service.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── model_utils.py
│   ├── dataset/
│   └── saved_models/
├── docker-compose.yml
├── Dockerfile.frontend
├── Dockerfile.backend
└── README.md
```

## Backend Endpoints

- `POST /upload-sample` - Upload training images for a class
- `POST /train` - Train the model on uploaded dataset
- `POST /predict` - Predict class from a single image
- `GET /dataset-stats` - Get current dataset statistics
- `GET /health` - Backend health check

## Setup in VS Code

### 1. Create a Python virtual environment

```powershell
cd c:\Users\Ahsan-Home\Desktop\PROJECT\FRONTEND
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd ../frontend
pip install -r requirements.txt
```

### 4. Launch backend

```powershell
cd ../backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Launch frontend

```powershell
cd ../frontend
streamlit run app.py
```

## Docker Setup

### Build and run with Docker Compose

```powershell
docker compose up --build
```

After startup:
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

## Notes

- The backend stores uploaded samples in `backend/dataset`.
- Trained model artifacts are saved in `backend/saved_models`.
- The frontend communicates with the backend through HTTP requests.
- The app is designed with a dark-themed dashboard and real-time status updates.
