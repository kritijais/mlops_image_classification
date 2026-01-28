import pickle
import torch
from torchvision import transforms
from src.model.cnn import SimpleCNN
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model = SimpleCNN()
    model.load_state_dict(payload["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, payload["classes"]

def predict_image(model, image: Image.Image, classes):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {
        "predicted_label": classes[int(probs.argmax())],
        "probabilities": {
            classes[0]: float(probs[0]),
            classes[1]: float(probs[1])
        }
    }
