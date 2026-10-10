# Refund Item Image Classifier - Batch Processing Prototype

A small, reproducible end-to-end prototype for the From Model to Production Environemnt oral project report: train a lightweight image classifier, package it behind a REST API, send batches of Base64-serialized images, write prediction probabilities to CSV, archive successfully processed images, and run the workflow nightly with Windows Task Scheduler.

**Important limitation:** this is an educational prototype using Fashion-MNIST (28x28 grayscale article images), not a production-ready classifier for arbitrary customer photos. Real deployment would require representative, consented/lawfully acquired, labeled images from the actual refund process, stronger image preprocessing, monitoring, security, and human review for uncertain predictions.

## What is in this repository?

- `data_utils.py` - downloads and parses the official Fashion-MNIST IDX files without requiring TensorFlow.
- `train_model.py` - trains a scikit-learn logistic classifier and writes accuracy, class metrics, a confusion matrix, and a serialized model artifact.
- `app.py` - Flask REST API with `GET /health` and `POST /predict/batch`.
- `make_demo_batch.py` - creates sample PNGs so the full workflow can be demonstrated without sourcing photos manually.
- `batch_predict.py` - reads all new images, calls the REST API in batches, writes per-class probabilities to CSV, and moves successful inputs to a dated processed folder.
- `run_nightly.ps1` - starts the API if needed and runs the batch job; use this with Windows Task Scheduler.
- `test_service.py` - small API smoke test.
- `architecture.mmd` - editable Mermaid architecture diagram source.

## 1. Environment setup (Windows 10/11)

Use Python 3.11 (64-bit) for the simplest compatibility path. In PowerShell, change to the folder containing this README:

```powershell
cd "C:\PATH\TO\refund_batch_classifier"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, the project's Python executable can still be called directly as `.\.venv\Scripts\python.exe`; alternatively, for the current terminal only, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## 2. Train and evaluate the model

```powershell
python train_model.py
```

The first run downloads the public Fashion-MNIST dataset (about 30 MB total) into `data/raw`. It trains on a reproducibly selected 30,000-image subset by default to keep the exercise manageable; the 10,000 official test images are held out from training.

Outputs:

- `artifacts/fashion_mnist_classifier.joblib` - model artifact loaded by the API
- `reports/metrics.json` - exact accuracy, macro precision/recall/F1, data size, model metadata
- `reports/classification_report.txt` - per-class precision, recall, F1 and support
- `reports/confusion_matrix.csv` and `reports/confusion_matrix.png` - error analysis


## 3. Create demo refund items

```powershell
python make_demo_batch.py --count 20
```

This saves 20 known Fashion-MNIST test images to `data/incoming` and a small ground-truth file to `data/demo_ground_truth.csv`. The images are known examples to test the plumbing, not realistic independent customer-uploaded photographs.

## 4. Start the REST API

Open **Terminal 1** and keep it running:

```powershell
python app.py
```

API base URL: `http://127.0.0.1:5000`

Check health in **Terminal 2**:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Expected fields include `status: ok`, `model_loaded: true`, `model_type`, and `model_version`. If it reports the model is missing, run the training step first.

## 5. Run an end-to-end batch and test the API

In Terminal 2, while the API remains running:

```powershell
python test_service.py
python batch_predict.py
```

The batch runner:

1. Looks for images in `data/incoming`.
2. Encodes each image's raw bytes as Base64 inside JSON.
3. Sends up to 16 images per HTTP request to `POST /predict/batch`.
4. Receives a predicted class and ten class probabilities per image.
5. Writes a dated CSV to `outputs`.
6. Moves successfully processed images to `data/processed/YYYY-MM-DD` so they are not picked up in the next nightly run.
7. Logs batch outcomes to `logs/batch.log`.

To repeat the demonstration after files were moved, run `python make_demo_batch.py --count 20` again.

## 6. Schedule nightly processing with Windows Task Scheduler

The REST API must be reachable when the job runs. The supplied script starts it in the background if it is not healthy.

1. Confirm training has created `artifacts/fashion_mnist_classifier.joblib`.
2. Search Windows for **Task Scheduler**, then choose **Create Basic Task**.
3. Name it `Refund image nightly batch`.
4. Trigger: **Daily**, start at **02:00** (or the desired overnight time).
5. Action: **Start a program**.
6. Program/script: `powershell.exe`.
7. Add arguments (replace the path with the real extracted folder path):

   ```text
   -NoProfile -ExecutionPolicy Bypass -File "C:\PATH\TO\refund_batch_classifier\run_nightly.ps1"
   ```

8. If the dialog has **Start in**, set it to the project folder, without quotes.
9. Test manually by right-clicking the task and choosing **Run**. Inspect `logs/nightly.log`, `logs/batch.log`, `outputs`, and `data/processed`.



## Monitoring / operational safeguards

For this prototype, monitor API health (`/health`), service stderr/stdout, `logs/batch.log`, `logs/nightly.log`, batch latency, number of images processed, per-file failures, and whether an output CSV was created. Never silently discard corrupt images. The prototype retains failed files in the incoming folder so they can be investigated. A real service should add alerting, retries with limits, secure authentication/TLS, request-size limits, access controls, a quarantine folder, model version registry/rollback, drift checks, and human review for low-confidence predictions.

