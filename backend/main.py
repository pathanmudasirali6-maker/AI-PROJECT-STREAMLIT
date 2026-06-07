import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.api import router
from backend.utils.model_utils import ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend")

app = FastAPI(
    title="AI Teachable Machine Backend",
    description="FastAPI backend for dataset uploads, training, and prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Ensuring dataset and model directories are available.")
    ensure_directories()

app.include_router(router)


@app.get("/health")
def health_check():
    """Simple health endpoint used by the frontend to verify backend availability."""
    logger.info("Health check requested")
    return {"status": "healthy"}
