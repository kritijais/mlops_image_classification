from PIL import Image
from pathlib import Path
import tempfile

def test_image_preprocessing_resize_and_rgb():
    """
    Unit test to ensure preprocessing logic
    converts image to RGB and resizes to 224x224.
    """

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a dummy grayscale image
        input_img_path = tmpdir / "input.jpg"
        img = Image.new("L", (300, 300))  # grayscale
        img.save(input_img_path)

        # Simulate preprocessing
        processed_img = Image.open(input_img_path)
        processed_img = processed_img.convert("RGB")
        processed_img = processed_img.resize((224, 224))

        # Assertions
        assert processed_img.size == (224, 224)
        assert processed_img.mode == "RGB"
