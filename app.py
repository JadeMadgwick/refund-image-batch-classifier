"""Flask REST API for single-request, multi-image predictions."""
from __future__ import annotations

import base64
import binascii
import io
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fashion_mnist_classifier.joblib"
MAX_BATCH_SIZE = 50
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # Keep requests bounded for the prototype.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BUNDLE = None
MODEL_LOAD_ERROR = None
try:
    if MODEL_PATH.exists():
        BUNDLE = joblib.load(MODEL_PATH)
    else:
        MODEL_LOAD_ERROR = f"Model not found at {MODEL_PATH}. Run: python train_model.py"
except Exception as exc:  # API starts, but health endpoint reports the model as unavailable.
    MODEL_LOAD_ERROR = f"Could not load model artifact: {type(exc).__name__}: {exc}"


def decode_image(encoded: str) -> np.ndarray:
    """Decode base64-serialized PNG/JPEG/etc. to a normalized 784-element vector."""
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 must be a non-empty base64 string")
    # Allow a data URI prefix as well as the raw base64 payload.
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("decoded bytes are not a supported image") from exc
    return pixels.reshape(1, 784)


@app.get("/health")
def health():
    if BUNDLE is None:
        return jsonify({"status": "unavailable", "model_loaded": False, "detail": MODEL_LOAD_ERROR}), 503
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_type": BUNDLE.get("model_type", "unknown"),
        "model_version": BUNDLE.get("trained_at_utc", "unknown"),
    }), 200


@app.post("/predict/batch")
def predict_batch():
    if BUNDLE is None:
        return jsonify({"error": "model_unavailable", "detail": MODEL_LOAD_ERROR}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        return jsonify({"error": "invalid_request", "detail": "Send JSON with an 'images' list."}), 400
    images = payload["images"]
    if not images:
        return jsonify({"error": "empty_batch", "detail": "The images list must not be empty."}), 400
    if len(images) > MAX_BATCH_SIZE:
        return jsonify({"error": "batch_too_large", "detail": f"Maximum is {MAX_BATCH_SIZE} images per request."}), 413

    start = time.perf_counter()
    results = []
    valid_vectors = []
    valid_positions = []
    for position, item in enumerate(images):
        item_id = item.get("id", f"item_{position}") if isinstance(item, dict) else f"item_{position}"
        if not isinstance(item, dict):
            results.append({"id": str(item_id), "error": "Each image entry must be an object."})
            continue
        try:
            vector = decode_image(item.get("image_base64"))
            valid_vectors.append(vector[0])
            valid_positions.append((position, str(item_id)))
            results.append(None)
        except ValueError as exc:
            results.append({"id": str(item_id), "error": str(exc)})

    if valid_vectors:
        probability_matrix = BUNDLE["model"].predict_proba(np.asarray(valid_vectors, dtype=np.float32))
        class_names = BUNDLE["class_names"]
        model_classes = BUNDLE["model"].classes_
        for row_num, (position, item_id) in enumerate(valid_positions):
            probs = probability_matrix[row_num]
            class_probabilities = {
                class_names[int(class_id)]: round(float(prob), 6)
                for class_id, prob in zip(model_classes, probs)
            }
            best_column = int(np.argmax(probs))
            best_class_id = int(model_classes[best_column])
            results[position] = {
                "id": item_id,
                "predicted_class": class_names[best_class_id],
                "confidence": round(float(probs[best_column]), 6),
                "probabilities": class_probabilities,
            }

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    errors = sum(1 for result in results if result and "error" in result)
    logging.info("batch_prediction count=%d errors=%d latency_ms=%.2f", len(images), errors, elapsed_ms)
    return jsonify({
        "count": len(images),
        "successful": len(images) - errors,
        "failed": errors,
        "latency_ms": elapsed_ms,
        "model_version": BUNDLE.get("trained_at_utc", "unknown"),
        "predictions": results,
    }), 200


if __name__ == "__main__":
    # Local demonstration only. Do not expose Flask's development server to the public internet.
    app.run(host="127.0.0.1", port=5000, debug=False)
