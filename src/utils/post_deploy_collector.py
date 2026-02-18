import os
import requests
from pathlib import Path

def collect_post_deployment_data():
    test_dir = Path("data/processed_split/test")
    cats_dir = test_dir / "cats"
    dogs_dir = test_dir / "dogs"

    results = []

    cat_images = list(cats_dir.glob("*.jpg"))[:5]  # Take first 5 cat images
    for cat_image in cat_images:
        response = requests.post("http://localhost:8000/predict", files={"file": open(cat_image, "rb")})
        results.append({
            "image_path": str(cat_image),
            "true_label": "cats",
            "predicted_label": response.json()["predicted_label"],
            "probabilities": response.json()["probabilities"]
        })

    dog_images = list(dogs_dir.glob("*.jpg"))[:5]  # Take first 5 dog images
    for dog_image in dog_images:
        response = requests.post("http://localhost:8000/predict", files={"file": open(dog_image, "rb")})
        results.append({
            "image_path": str(dog_image),
            "true_label": "dogs",
            "predicted_label": response.json()["predicted_label"],
            "probabilities": response.json()["probabilities"]
        })

    return results

if __name__ == "__main__":
    results = collect_post_deployment_data()
    for result in results:
        print(result)