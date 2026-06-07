import io
import uuid
from pathlib import Path

from PIL import Image

from backend.utils.model_utils import DATASET_DIR


def get_image_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def save_upload_files(class_name: str, upload_file) -> Path:
    class_dir = DATASET_DIR / class_name.strip()
    class_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(upload_file.filename).suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{extension}"
    target_path = class_dir / filename

    upload_file.file.seek(0)
    content = upload_file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(io.BytesIO(content)).convert("RGB")
    image.save(target_path)
    return target_path
