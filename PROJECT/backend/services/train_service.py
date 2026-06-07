import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from backend.utils.model_utils import get_paths

logger = logging.getLogger("backend.train")


def get_image_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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


def get_dataset_stats() -> dict:
    paths = get_paths()
    dataset_dir = paths["dataset_dir"]
    class_directories = [p for p in dataset_dir.iterdir() if p.is_dir()]
    total_images = 0
    for class_dir in class_directories:
        total_images += len([f for f in class_dir.iterdir() if f.is_file()])
    return {
        "total_classes": len(class_directories),
        "total_images": total_images,
    }


def load_image_tensor(image_path: Path):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    transform = get_image_transform()
    return transform(image)


def extract_features(image_tensors, extractor) -> np.ndarray:
    import torch

    with torch.no_grad():
        features = extractor(image_tensors)
        features = features.view(features.size(0), -1)
        return features.cpu().numpy()


def train_model() -> dict:
    import torch

    paths = get_paths()
    dataset_dir = paths["dataset_dir"]
    model_file = paths["model_file"]
    label_file = paths["label_file"]

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError("No dataset classes found. Upload sample images before training.")

    tensors = []
    labels = []
    for class_dir in class_dirs:
        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue
            try:
                tensors.append(load_image_tensor(image_path))
                labels.append(class_dir.name)
            except Exception as exc:
                logger.warning("Skipping invalid image %s: %s", image_path, exc)

    if not tensors:
        raise ValueError("Dataset contains no valid images.")

    batch_tensor = torch.stack(tensors)
    extractor = load_feature_extractor()
    features = extract_features(batch_tensor, extractor)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )
    classifier.fit(features, encoded_labels)

    joblib.dump(classifier, model_file)
    joblib.dump(encoder, label_file)

    return get_dataset_stats()
