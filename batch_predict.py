
from __future__ import annotations

import argparse
import base64
import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly batch predictions through the Flask service.")
    parser.add_argument("--input", default=str(ROOT / "data" / "incoming"), help="Folder holding unprocessed image files.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"), help="Folder where prediction CSVs are saved.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"), help="Folder where successfully processed files are moved.")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000", help="Base URL for the REST API.")
    parser.add_argument("--batch-size", type=int, default=16, help="Images sent per API request, maximum 50.")
    parser.add_argument("--no-move", action="store_true", help="Leave input images in place (handy for debugging).")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        parser.error("--batch-size must be between 1 and 50")

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    processed_root = Path(args.processed_dir).resolve()
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "batch.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8"
    )

    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        return 2
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        print(f"No new images found in {input_dir}. Nothing to process.")
        logging.info("batch_run found zero files")
        return 0

    # Check the service before reading/posting image payloads.
    try:
        health = requests.get(f"{args.api_url.rstrip('/')}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"API is not healthy at {args.api_url}: {exc}")
        logging.exception("batch_run aborted: API health check failed")
        return 2

    records = []
    successful_names = set()
    for group in chunks(files, args.batch_size):
        payload_images = []
        for path in group:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            payload_images.append({"id": path.name, "image_base64": encoded})
        try:
            response = requests.post(
                f"{args.api_url.rstrip('/')}/predict/batch",
                json={"images": payload_images}, timeout=120
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"API call failed; files were not moved. Error: {exc}")
            logging.exception("batch_run API call failed")
            return 2

        for item in body.get("predictions", []):
            row = {"file_name": item.get("id", ""), "status": "ok" if "error" not in item else "error"}
            if "error" in item:
                row.update({"predicted_class": "", "confidence": "", "error": item["error"]})
            else:
                row.update({
                    "predicted_class": item["predicted_class"],
                    "confidence": item["confidence"],
                    "error": "",
                })
                successful_names.add(item["id"])
            probabilities = item.get("probabilities", {})
            for class_name in CLASS_NAMES:
                row[f"probability_{safe_column(class_name)}"] = probabilities.get(class_name, "")
            records.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"predictions_{timestamp}.csv"
    fieldnames = ["file_name", "status", "predicted_class", "confidence", "error"] + [
        f"probability_{safe_column(name)}" for name in CLASS_NAMES
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if not args.no_move:
        processed_dir = processed_root / datetime.now().strftime("%Y-%m-%d")
        processed_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            if path.name in successful_names and path.exists():
                destination = processed_dir / path.name
                if destination.exists():
                    destination = processed_dir / f"{path.stem}_{timestamp}{path.suffix}"
                shutil.move(str(path), str(destination))

    failed = sum(1 for row in records if row["status"] != "ok")
    print(f"Processed: {len(records)} image(s); successful: {len(records)-failed}; failed: {failed}")
    print(f"Predictions saved to: {output_path}")
    logging.info("batch_run total=%d successful=%d failed=%d csv=%s", len(records), len(records)-failed, failed, output_path)
    return 1 if failed else 0


def safe_column(text: str) -> str:
    return text.lower().replace("/", "_").replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    raise SystemExit(main())
