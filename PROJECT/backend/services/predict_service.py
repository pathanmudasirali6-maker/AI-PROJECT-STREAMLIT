import io

import joblib
import numpy as np
from PIL import Image

from backend.utils.model_utils import get_paths


def load_feature_extractor():
    import torch
    from torchvision import models

    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        base_model = models.mobilenet_v3_small(pretrained=True)

    base_model.eval()
    extractor = torch.nn.Sequential(base_model.features, base_model.avgpool)
    return extractor


def load_saved_pipeline() -> tuple:
    paths = get_paths()
    model_file = paths["model_file"]
    label_file = paths["label_file"]
    if not model_file.exists() or not label_file.exists():
        raise FileNotFoundError("Saved model files were not found. Train the model first.")

    classifier = joblib.load(model_file)
    label_encoder = joblib.load(label_file)
    return classifier, label_encoder


def get_image_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def prepare_image_tensor(image_bytes: bytes):
    import torch

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_image_transform()
    return transform(image).unsqueeze(0)


def predict_image(image_bytes: bytes) -> dict:
    import torch

    classifier, label_encoder = load_saved_pipeline()
    extractor = load_feature_extractor()
    input_tensor = prepare_image_tensor(image_bytes)
    with torch.no_grad():
        features = extractor(input_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()

    probabilities = classifier.predict_proba(features)[0]
    chosen_index = int(np.argmax(probabilities))
    return {
        "label": label_encoder.inverse_transform([chosen_index])[0],
        "confidence": float(probabilities[chosen_index]),
        "probabilities": {
            label_encoder.inverse_transform([idx])[0]: float(prob)
            for idx, prob in enumerate(probabilities)
        },
    }
