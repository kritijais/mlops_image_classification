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

    record_request(
        path=request.url.path,
        latency=latency,
        status_code=response.status_code
    )

    return response

# -----------------------------
# Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    output = {}
    for path in REQUEST_COUNT:
        count = REQUEST_COUNT[path]
        avg_latency = TOTAL_LATENCY[path] / max(count, 1)

        output[path] = {
            "request_count": count,
            "avg_latency_ms": round(avg_latency * 1000, 2)
        }
    return output
