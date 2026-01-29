from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image
import io
import time

from src.model.inference import load_model, predict_image
from src.utils.monitoring import record_request, REQUEST_COUNT, TOTAL_LATENCY


MODEL_PATH = "artifacts/cnn_model.pkl"

app = FastAPI(
    title="Cats vs Dogs CNN Inference API",
    version="1.0.0"
)

model, classes = load_model(MODEL_PATH)

PROM_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method"]
)

PROM_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["path"]
)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(model, image, classes)
    return result

# -----------------------------
# Monitoring Middleware
# -----------------------------
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    # Custom monitoring
    record_request(
        path=request.url.path,
        latency=latency,
        status_code=response.status_code
    )

    # Prometheus metrics
    PROM_REQUEST_COUNT.labels(
        path=request.url.path,
        method=request.method
    ).inc()

    PROM_REQUEST_LATENCY.labels(
        path=request.url.path
    ).observe(latency)

    return response

# -----------------------------
# Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )