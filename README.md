# AI-PROJECT-STREAMLIT

A Streamlit frontend paired with a FastAPI backend for building a teachable image classifier.

## Run locally

1. Start the backend:

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend:

```bash
cd frontend
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Run with Docker Compose

From the repository root:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

## GitHub Deployment

A GitHub Actions workflow is included at `.github/workflows/ci-deploy.yml`.

It performs:

- Python syntax validation for `backend` and `frontend`
- Docker image builds for the backend and frontend
- Publish to GitHub Container Registry on pushes to `main`

### Publishing images

The workflow pushes images as:

- `ghcr.io/${{ github.repository }}/ai-project-streamlit-backend:latest`
- `ghcr.io/${{ github.repository }}/ai-project-streamlit-frontend:latest`

No extra secrets are required beyond the default `GITHUB_TOKEN`.
