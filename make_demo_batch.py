"""Create a small repeatable set of demo input PNGs from the official test split."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from data_utils import load_fashion_mnist

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def safe_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo PNGs in data/incoming.")
    parser.add_argument("--count", type=int, default=20, help="Number of demo images (default: 20).")
    args = parser.parse_args()
    if not 1 <= args.count <= 200:
        parser.error("--count must be between 1 and 200")

    incoming = ROOT / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    for old in incoming.glob("demo_*.png"):
        old.unlink()

    (_, _), (x_test, y_test) = load_fashion_mnist()
    rng = np.random.default_rng(17)
    # Choose a random selection from the held-out official test set.
    selected = rng.choice(len(x_test), size=min(args.count, len(x_test)), replace=False)
    rows = []
    for i, index in enumerate(selected):
        label = int(y_test[index])
        filename = f"demo_{i:03d}_true-{safe_label(CLASS_NAMES[label])}.png"
        Image.fromarray(x_test[index], mode="L").save(incoming / filename)
        rows.append({"file_name": filename, "true_class": CLASS_NAMES[label], "test_set_index": int(index)})

    with open(ROOT / "data" / "demo_ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "true_class", "test_set_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} demo image(s) in: {incoming}")
    print("Ground truth is in data/demo_ground_truth.csv. These images are for a controlled demo, not independent real-world photos.")


if __name__ == "__main__":
    main()
