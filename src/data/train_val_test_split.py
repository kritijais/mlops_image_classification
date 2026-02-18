import os
import random
import shutil
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
SOURCE_DIR = Path("data/processed")
TARGET_DIR = Path("data/processed_split")

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# -----------------------------
# Utility Functions
# -----------------------------
def get_images(directory):
    return [
        f for f in directory.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ]

# -----------------------------
# Main Split Function
# -----------------------------
def split_dataset():
    random.seed(RANDOM_SEED)

    for cls in ["cats", "dogs"]:
        src_cls_dir = SOURCE_DIR / cls
        images = get_images(src_cls_dir)

        random.shuffle(images)
        total = len(images)

        train_end = int(SPLIT_RATIOS["train"] * total)
        val_end = train_end + int(SPLIT_RATIOS["val"] * total)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in splits.items():
            dest_dir = TARGET_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy(img_path, dest_dir / img_path.name)

        print(
            f"{cls}: "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

    print("Dataset split completed successfully.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    split_dataset()
