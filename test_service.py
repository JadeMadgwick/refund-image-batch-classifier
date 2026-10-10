"""Lightweight API contract checks. Run: python test_service.py (after training)."""
import base64
import io

from PIL import Image

import app as service


def main():
    client = service.app.test_client()
    health = client.get("/health")
    print("GET /health:", health.status_code, health.get_json())
    if health.status_code != 200:
        raise SystemExit("Health check failed. Train the model first: python train_model.py")

    image = Image.new("L", (28, 28), color=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    response = client.post("/predict/batch", json={"images": [{"id": "smoke-test.png", "image_base64": encoded}]})
    print("POST /predict/batch:", response.status_code, response.get_json())
    if response.status_code != 200:
        raise SystemExit("Prediction endpoint did not return HTTP 200")
    body = response.get_json()
    first = body["predictions"][0]
    if "predicted_class" not in first or "probabilities" not in first:
        raise SystemExit("Prediction response is missing the class or probabilities")
    if abs(sum(first["probabilities"].values()) - 1.0) > 0.02:
        raise SystemExit("Returned probabilities do not approximately sum to 1.0")
    print("PASS: service contract returned a class and probabilities.")


if __name__ == "__main__":
    main()
