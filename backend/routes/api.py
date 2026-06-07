import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.predict_service import predict_image
from backend.services.train_service import get_dataset_stats, train_model
from backend.utils.image_utils import save_upload_files

logger = logging.getLogger("backend.api")
router = APIRouter(tags=["teachable-machine"])

@router.post("/upload-sample")
async def upload_sample(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    logger.info("Upload sample request received: class=%s files=%d", class_name, len(files))
    if not class_name.strip():
        raise HTTPException(status_code=400, detail="Class name is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    saved_count = 0
    for upload_file in files:
        try:
            upload_file.file.seek(0)
            save_upload_files(class_name, upload_file)
            saved_count += 1
        except Exception as exc:
            logger.error("Failed to save image %s: %s", upload_file.filename, exc)
            raise HTTPException(status_code=500, detail="Unable to save one or more files.")

    stats = get_dataset_stats()
    return {
        "message": f"Uploaded {saved_count} sample(s) for class '{class_name}'.",
        "total_classes": stats["total_classes"],
        "total_images": stats["total_images"],
    }

@router.post("/train")
def train():
    logger.info("Train request received")
    try:
        stats = train_model()
        return {
            "status": "success",
            "message": "Training finished successfully.",
            "total_classes": stats["total_classes"],
            "total_images": stats["total_images"],
        }
    except Exception as exc:
        logger.exception("Training error")
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    logger.info("Prediction request received: filename=%s content_type=%s", image.filename, image.content_type)
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Image file must be PNG or JPEG.")
    try:
        content = await image.read()
        result = predict_image(content)
        return {
            "predicted_class": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }
    except FileNotFoundError as exc:
        logger.exception("Model artifacts missing")
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/dataset-stats")
def dataset_stats():
    logger.info("Dataset statistics requested")
    return get_dataset_stats()
