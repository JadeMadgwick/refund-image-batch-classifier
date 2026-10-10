"""Download and read Fashion-MNIST IDX gzip files from the official Keras mirror.

Fashion-MNIST is a Zalando Research dataset. This helper avoids a TensorFlow
installation, which keeps the prototype lightweight and quick to reproduce.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = BASE_URL + filename
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {filename} from the Fashion-MNIST mirror...")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "RefundBatchClassifier/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file: {filename}")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {filename}. Check your internet connection and retry. URL: {url}"
        )
    return destination


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid image file header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected image-file magic number {magic} in {path}")
        raw = stream.read()
    expected = count * rows * cols
    images = np.frombuffer(raw, dtype=np.uint8)
    if images.size != expected:
        raise ValueError(f"Expected {expected} pixels but found {images.size} in {path}")
    return images.reshape(count, rows, cols).copy()


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid label file header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected label-file magic number {magic} in {path}")
        raw = stream.read()
    labels = np.frombuffer(raw, dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Expected {count} labels but found {labels.size} in {path}")
    return labels.copy()


def load_fashion_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return official train/test arrays: images uint8 (N, 28, 28), labels 0-9."""
    paths = {key: _download_if_missing(filename) for key, filename in FILES.items()}
    x_train = _read_images(paths["train_images"])
    y_train = _read_labels(paths["train_labels"])
    x_test = _read_images(paths["test_images"])
    y_test = _read_labels(paths["test_labels"])

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Dataset integrity check failed: image/label counts differ.")
    if x_train.shape[1:] != (28, 28) or x_test.shape[1:] != (28, 28):
        raise ValueError("Dataset integrity check failed: expected 28x28 images.")
    return (x_train, y_train), (x_test, y_test)
