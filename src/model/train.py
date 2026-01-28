import torch
import pickle
import mlflow
from torch import nn, optim
from sklearn.metrics import accuracy_score
from src.model.cnn import SimpleCNN
from src.model.dataset import get_dataloaders
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_loss_curves
)

import os

if os.getenv("CI") == "true":
    print("CI environment detected — skipping training.")
    exit(0)

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/processed_split"
MODEL_PATH = "artifacts/cnn_model.pkl"
PLOT_DIR = "artifacts/plots"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# -----------------------------
# Training Function
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_experiment("cats_vs_dogs_cnn_augmented")

    train_losses, val_losses = [], []
    final_y_true, final_y_pred = [], []

    with mlflow.start_run():

        # -----------------------------
        # Log Parameters
        # -----------------------------
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param(
            "augmentation",
            "RandomCrop, Flip, Rotation, ColorJitter"
        )

        for epoch in range(EPOCHS):

            # -------- TRAIN --------
            model.train()
            running_train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # -------- VALIDATE --------
            model.eval()
            running_val_loss = 0.0
            y_true, y_pred = [], []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x)
                    loss = criterion(preds, y)

                    running_val_loss += loss.item()
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.argmax(dim=1).cpu().numpy())

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            val_acc = accuracy_score(y_true, y_pred)

            # -----------------------------
            # Log Metrics
            # -----------------------------
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            final_y_true = y_true
            final_y_pred = y_pred

        # -----------------------------
        # Save Model Artifact
        # -----------------------------
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {"model_state": model.state_dict(), "classes": classes},
                f
            )

        mlflow.log_artifact(MODEL_PATH)

        # -----------------------------
        # Create & Log Plots
        # -----------------------------
        cm_path = f"{PLOT_DIR}/confusion_matrix.png"
        loss_curve_path = f"{PLOT_DIR}/loss_curves.png"

        plot_confusion_matrix(
            final_y_true, final_y_pred, classes, cm_path
        )
        plot_loss_curves(train_losses, val_losses, loss_curve_path)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(loss_curve_path)

        print("MLflow run completed successfully.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
