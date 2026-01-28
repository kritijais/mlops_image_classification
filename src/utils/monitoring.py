import time
import logging
from collections import defaultdict

# -----------------------------
# Logger configuration
# -----------------------------
logger = logging.getLogger("inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------
# In-app metrics
# -----------------------------
REQUEST_COUNT = defaultdict(int)
TOTAL_LATENCY = defaultdict(float)

def record_request(path: str, latency: float, status_code: int):
    REQUEST_COUNT[path] += 1
    TOTAL_LATENCY[path] += latency

    logger.info(
        f"path={path} "
        f"status={status_code} "
        f"latency_ms={latency * 1000:.2f}"
    )
