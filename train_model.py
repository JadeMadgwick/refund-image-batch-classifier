"""Train and evaluate a lightweight Fashion-MNIST image classifier."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_utils import load_fashion_mnist

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the refund image classifier.")
    parser.add_argument("--samples", type=int, default=30000,
                        help="Number of training images to use (default: 30000; max: 60000).")
    parser.add_argument("--iterations", type=int, default=35,
                        help="Maximum SGD training iterations (default: 35).")
    args = parser.parse_args()
    if args.samples < 1000 or args.samples > 60000:
        parser.error("--samples must be between 1000 and 60000")
    if args.iterations < 1 or args.iterations > 300:
        parser.error("--iterations must be between 1 and 300")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x_train), size=min(args.samples, len(x_train)), replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    print(f"2/5 Preparing {len(x_train):,} training and {len(x_test):,} test images...")
    # Flatten each 28x28 image into 784 features and scale pixels from 0-255 to 0-1.
    X_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    X_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    print("3/5 Training a multinomial logistic classifier with SGD...")
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=args.iterations,
        tol=1e-3,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        learning_rate="optimal",
    )
    model.fit(X_train, y_train)

    print("4/5 Evaluating on the held-out test split...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(
        y_test, y_pred, labels=np.arange(10), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, labels=np.arange(10), target_names=CLASS_NAMES,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
    cm_csv = REPORTS_DIR / "confusion_matrix.csv"
    np.savetxt(cm_csv, cm, delimiter=",", fmt="%d", header=",".join(CLASS_NAMES), comments="")
    (REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    metrics = {
        "dataset": "Fashion-MNIST (Zalando Research)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "image_shape": [28, 28],
        "model_type": "SGDClassifier(loss='log_loss')",
        "iterations_requested": args.iterations,
        "iterations_run": int(getattr(model, "n_iter_", 0)),
        "accuracy": accuracy,
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "class_report": report_dict,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save a visual confusion matrix for the presentation/report.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Fashion-MNIST test-set confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model": model,
        "class_names": CLASS_NAMES,
        "dataset": "Fashion-MNIST",
        "image_shape": [28, 28],
        "trained_at_utc": metrics["trained_at_utc"],
        "training_samples": len(x_train),
        "test_accuracy": accuracy,
        "model_type": metrics["model_type"],
    }
    artifact_path = ARTIFACTS_DIR / "fashion_mnist_classifier.joblib"
    joblib.dump(bundle, artifact_path, compress=3)

    print("5/5 Training complete.")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print("\nPer-class report:\n")
    print(report_text)
    print(f"Model artifact: {artifact_path}")
    print(f"Metrics JSON:   {REPORTS_DIR / 'metrics.json'}")
    print(f"Confusion plot: {REPORTS_DIR / 'confusion_matrix.png'}")
    print("Copy the printed accuracy and macro-F1 into the presentation; do not guess them.")


if __name__ == "__main__":
    main()
