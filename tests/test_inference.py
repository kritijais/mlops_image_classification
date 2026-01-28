import torch
from PIL import Image
from src.model.inference import predict_image

class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Always return fixed logits
        return torch.tensor([[1.0, 2.0]])

def test_predict_image_output_format():
    """
    Unit test for inference utility.
    Ensures output structure and probability correctness.
    """

    model = DummyModel()
    classes = ["cats", "dogs"]

    # Create a dummy RGB image
    image = Image.new("RGB", (224, 224))

    result = predict_image(model, image, classes)

    # Assertions
    assert "predicted_label" in result
    assert "probabilities" in result

    probs = result["probabilities"]

    assert set(probs.keys()) == set(classes)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert result["predicted_label"] in classes
