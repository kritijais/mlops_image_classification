import kagglehub
import shutil
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset to kagglehub cache
dataset_path = Path(
    kagglehub.dataset_download(
        "bhavikjikadara/dog-and-cat-classification-dataset"
    )
)

print("Downloaded dataset path:", dataset_path)

# Handle dataset structure (PetImages/Cat, PetImages/Dog)
# Find the directory that contains Cat and Dog
for root in dataset_path.iterdir():
    if root.is_dir() and {"Cat", "Dog"}.issubset({p.name for p in root.iterdir() if p.is_dir()}):
        source_root = root
        break
else:
    raise RuntimeError("Could not find Cat and Dog directories in dataset")

# Copy Cat and Dog folders directly into data/raw
for class_name in ["Cat", "Dog"]:
    src = source_root / class_name
    dst = RAW_DATA_DIR / class_name

    shutil.copytree(src, dst, dirs_exist_ok=True)

print("Dataset staged to:", RAW_DATA_DIR.resolve())
