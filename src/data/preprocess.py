import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_MAPPING = {
    "Cat": "cats",
    "Dog": "dogs"
}

# -----------------------------
# Core Preprocessing Function
# -----------------------------
def preprocess_dataset():
    """
    Converts raw Kaggle Cats vs Dogs images into
    224x224 RGB images suitable for CNN training.
    """
    for raw_class, norm_class in CLASS_MAPPING.items():
        input_dir = RAW_DIR / raw_class
        output_dir = PROCESSED_DIR / norm_class
        output_dir.mkdir(parents=True, exist_ok=True)

        images = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ]

        print(f"Processing {len(images)} images for class '{norm_class}'")

        for img_path in tqdm(images):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)

                    out_path = output_dir / img_path.name
                    img.save(out_path, format="JPEG", quality=95)

            except Exception as e:
                print(f"[WARNING] Skipping {img_path.name}: {e}")

    print("Preprocessing complete.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    preprocess_dataset()
