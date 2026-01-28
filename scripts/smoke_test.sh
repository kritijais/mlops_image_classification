#!/bin/bash
set -e  # Exit immediately on error

API_URL="http://localhost:8000"

echo "Running smoke tests..."

# -----------------------------
# Health check
# -----------------------------
echo "Checking /health endpoint"
curl -f "${API_URL}/health"
echo "Health check passed"

# -----------------------------
# Prediction check
# -----------------------------
echo "Checking /predict endpoint"

TEST_IMAGE="tests/assets/sample_dog.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
  echo "Test image not found: $TEST_IMAGE"
  exit 1
fi

RESPONSE=$(curl -s -X POST "${API_URL}/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${TEST_IMAGE}")

echo "Prediction response: $RESPONSE"

# Basic validation
echo "$RESPONSE" | grep -q "predicted_label"
echo "$RESPONSE" | grep -q "probabilities"

echo "Prediction smoke test passed"

echo "All smoke tests passed successfully"
