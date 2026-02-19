# Cats vs Dogs Image Classification

## Project Overview

This project implements a **complete end-to-end MLOps pipeline** for **binary image classification (Cats vs Dogs)**, covering:

* Data preprocessing & splitting
* CNN model training with augmentation
* Experiment tracking using MLflow
* Data and Model artifact versioning with DVC
* Containerized inference service using FastAPI
* CI/CD automation with GitHub Actions
* Deployment using Docker Compose
* Post-deployment monitoring, smoke tests, and evaluation

---

## Dataset

* **Source:** Kaggle
* **Dataset:** Dog and Cat Classification Dataset
* **Classes:** Cats, Dogs
* **Images:** ~25,000

Raw images are standardized to **224×224 RGB** before training.

---


## Github URL

https://github.com/kritijais/mlops_image_classification

---


## Project Video URL

https://youtu.be/_VCGnujM4uw

---


## Project Architecture

```
Raw Data (Kaggle)
   ↓
Preprocessing (224×224 RGB)
   ↓
Train / Val / Test Split (80/10/10)
   ↓
CNN Training + Data Augmentation
   ↓
MLflow Experiment Tracking
   ↓
Serialized Model (.pkl)
   ↓
FastAPI Inference Service
   ↓
Docker Image
   ↓
GitHub Container Registry (GHCR)
   ↓
CI (Test + Build + Push)
   ↓
CD (Docker Compose Deployment)
   ↓
Smoke Tests
   ↓
Monitoring & Post-Deployment Evaluation
```

---

## Repository Structure

```
mlops_assignment2_image_classification/
│
├── data/
│   ├── raw/                     # Kaggle dataset (DVC-tracked)
│   ├── processed/               # 224x224 RGB images
│   └── processed_split/         # train/val/test split
│
├── src/
│   ├── data/
│   │   ├── data_download.py          # downloads dataset
│   │   └── preprocess.py             # standardizes raw images (size, color, format)
│   │   └── train_val_test_split.py   # organizes images into train/val/test sets 
│   │
│   ├── model/
│   │   ├── cnn.py                    # simple cnn model implementation
│   │   ├── dataset.py                # dataset loader
│   │   ├── train.py                  # train the model, save as .pkl and experiment tracking
│   │   └── inference.py              # inference
│   │
│   ├── api/
│   │   └── app.py                    # fast api inference service
│   │
│   └── utils/
│       ├── visualization.py          # confusion matrix, loss curves
│       ├── monitoring.py             # metrics - request count and latency
│       └── post_deploy_collector.py  # model performance tracking post deployment
│
├── tests/
│   ├── test_preprocess.py            # test one data preprocessing function
│   ├── test_inference.py             # test one model utility / inference function
│   └── assets/
│       └── sample_dog.jpg            # test image
│
├── scripts/
│   └── smoke_test.sh                 # post-deploy smoke test script
│
├── artifacts/
│   ├── cnn_model.pkl                 # saved model
│   └── plots/                        # visualization outputs
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── .gitignore
├── .dvcignore
└── README.md
```

---

## Key Technologies Used

| Category            | Tools                             |
| ------------------- | --------------------------------- |
| ML                  | PyTorch                           |
| Experiment Tracking | MLflow                            |
| Data Versioning     | DVC                               |
| API                 | FastAPI                           |
| Containerization    | Docker                            |
| Registry            | GitHub Container Registry (GHCR)  |
| CI/CD               | GitHub Actions                    |
| Deployment          | Docker Compose                    |
| Monitoring          | In-app logging & metrics          |
| Testing             | pytest                            |

---

## Installation Steps
Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
# or env\Scripts\activate on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## Dataset Download

The raw Cats vs Dogs dataset is downloaded directly from Kaggle using `kagglehub` and normalized into a consistent directory structure expected by the preprocessing pipeline.

### Download Raw Data

Run the following command from the project root:

```bash
python -m src.data.data_download
```

This command will:

* Download the dataset from Kaggle via kagglehub
* Extract and normalize the structure to:

```text
data/raw/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

---


### Data Preprocessing

Before training, all images undergo standardized preprocessing to ensure compatibility with CNN architectures:

* **Resize:** `224 × 224`
* **Color Mode:** RGB (3 channels)
* **Normalization:** ImageNet mean and standard deviation
* **Format:** Converted to PyTorch tensors

This preprocessing ensures:

* Uniform input dimensions
* Stable gradient behavior during training
* Compatibility with standard CNN backbones

---

### Data Augmentation (Training Only)

To improve generalization and reduce overfitting, the following augmentations are applied **only to the training set**:

* Random horizontal flip
* Random rotation
* Random resized crop
* Color jitter

Validation and test sets use **deterministic transforms only**, ensuring unbiased evaluation.

---

### Train / Validation / Test Split

The processed dataset is split using the following proportions:

| Split      | Percentage |
| ---------- | ---------- |
| Training   | 80%        |
| Validation | 10%        |
| Test       | 10%        |

The split is **class-balanced** and performed once to ensure reproducibility.

Resulting directory structure:

```text
data/processed_split/
├── train/
│   ├── cats/
│   └── dogs/
├── val/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

---

### End-to-End Data Preparation Flow

```bash
# Download raw data
python -m src.data.data_download

# Preprocess raw images
python -m src.data.preprocess

# Create train / val / test split
python -m src.data.train_val_test_split
```

---

### Data Versioning with DVC

After downloading the dataset, the raw data should be tracked using **DVC**:

This ensures:
* Raw data is reproducible
* Git history remains lightweight
* CI/CD pipelines stay data-agnostic

* All processed datasets and splits are tracked using **DVC**
* Git stores only lightweight `.dvc` metadata files
* Actual image data is stored in a DVC remote

```bash
dvc init
dvc add data/raw
dvc add data/processed
dvc add data/processed_split
```

### Configure DVC Remote Storage

**We are using local remote for this project but in real world we can use S3/Azure storage**

```bash
mkdir dvc_storage
dvc remote add -d storage dvc_storage
```

This creates:

```
.dvc/config
dvc_storage/
```

Config:

```ini
['core']
    remote = storage

['remote "storage"']
    url = dvc_storage
```

---

### Push data to DVC remote

```bash
dvc push
```

This uploads raw and processed datasets to the configured storage.

---

### Pull data

If datasets are missing:

```bash
dvc pull
```

This reconstructs:

```
data/raw
data/processed
data/processed_split
```

using the metadata files.

---

### CI/CD Integration

Before model training or inference, CI pipelines must restore data:

```bash
pip install dvc
dvc pull
```

This ensures pipelines remain **data-agnostic** while still reproducible.

---


## Model Training & Experiment Tracking

* **Model:** Simple CNN (baseline)
* **Input:** 224×224 RGB images
* **Loss:** Cross-Entropy
* **Optimizer:** Adam
* **Augmentations:**

  * Random crop
  * Horizontal flip
  * Rotation
  * Color jitter

### MLflow Logs:

* Parameters (epochs, batch size, learning rate)
* Metrics (train loss, validation loss, accuracy)
* Artifacts:

  * `cnn_model.pkl`
  * Confusion matrix
  * Loss curves

Run training:

```bash
python -m src.model.train
```

Launch MLflow UI:

```bash
mlflow ui
```

---

### Model Versioning with DVC

After generating the `.pkl` file, it should be tracked using **DVC`:

```bash
dvc add artifacts/cnn_model.pkl
```

---


## Inference Service

* **Framework:** FastAPI
* **Endpoints:**

  * `GET /health`
  * `POST /predict`
  * `GET /metrics`


### Run the API Locally

From project root:
```bash
uvicorn src.api.app:app --reload
```

Open:

http://127.0.0.1:8000/docs


### Example Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/assets/sample_dog.jpg"
```

---

## Containerization

Build image:

```bash
docker build -t cats-dogs-inference .
or
docker build -t ghcr.io/<github-username>/cats-dogs-inference:latest .
```

---

## Container Registry: GitHub Container Registry (GHCR)

Docker images are pushed to **GitHub Container Registry**:

```
ghcr.io/<github-username>/<repo-name>/cats-dogs-inference:latest
```

Authentication is handled automatically in CI using `GITHUB_TOKEN`.

---

## CI/CD Pipelines (GitHub Actions)

### Continuous Integration (CI)

Triggered on every **push / pull request**:

* Checkout repository
* Install dependencies
* Run unit tests (`pytest`)
* Build Docker image

### Continuous Deployment (CD)

Triggered on **push to `main`**:

* Pull latest image from **GHCR**
* Deploy/update service using **Docker Compose**
* Run post-deploy smoke tests

---

## Deployment with Docker Compose

### Deploy

```bash
docker compose up -d
```

The API will be accessible at:

http://localhost:8000

---

## Post-Deploy Smoke Tests

Automated smoke tests are run as part of the CD pipeline after deployment. These tests validate:

* `/health` endpoint
* `/predict` endpoint with a real image

To run smoke tests manually:
```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

If any check fails, **the CD pipeline fails automatically**.

---

## Monitoring & Logging

* Request count per endpoint
* Average latency tracking
* Structured logs (no images or sensitive data)
* `/metrics` endpoint for inspection

### Prometheus Monitoring

* Prometheus is configured to scrape metrics from the `/metrics` endpoint
* Metrics include request count, latency, and other performance indicators
* Prometheus is run as a separate service in the Docker Compose configuration

To access Prometheus:
1. Ensure Docker Compose is running (`docker compose up -d`)
2. Open Prometheus at http://localhost:9090

### Grafana Dashboards

* Grafana is used to visualize metrics collected by Prometheus
* Dashboards provide insights into API performance and health
* Grafana is configured with a provisioning file for datasources and dashboards

To access Grafana:
1. Ensure Docker Compose is running (`docker compose up -d`)
2. Open Grafana at http://localhost:3000

---

## Post-Deployment Performance Tracking

* Collect predictions + true labels (simulated or real)
* Stored in append-only JSONL
* Enables post-deployment accuracy evaluation

### Post-Deployment Performance Tracking

To evaluate model performance post-deployment:

1. Ensure the FastAPI service is running (`uvicorn src.api.app:app --reload`)
2. Run the post-deployment collector:
```bash
python -m src.utils.post_deploy_collector
```

This will send test images to the `/predict` endpoint and log the results.

---