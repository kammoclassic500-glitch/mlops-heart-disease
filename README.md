# Heart Disease Risk Prediction — MLOps Assignment (AIMLCZG523)

This repo is a working scaffold for every task in the assignment.
Every script in here has been run and tested — training produces a real model,
the API returns real predictions, and `pytest`/`flake8` both pass. What's left
for you is the parts that need *your* machine/cloud account: Docker build,
Kubernetes deployment, screenshots, GitHub repo, and the demo video.

## 0. One-time setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data Acquisition & EDA

```bash
python data/download_data.py     # pulls the official UCI dataset (id=45) -> data/heart_disease_raw.csv
python src/eda.py --data data/heart_disease_raw.csv
```

This produces, under `reports/eda/`:
- `class_balance.png` — bar chart of disease vs. no-disease counts
- `histograms.png` — distribution of each numeric feature
- `correlation_heatmap.png` — correlation matrix across all features + target
- `summary_stats.csv` — descriptive statistics

**Checklist**
- [ ] Ran `download_data.py`, confirmed `data/heart_disease_raw.csv` exists
- [ ] Ran `eda.py`, reviewed the 3 plots for anything unusual (outliers, imbalance)
- [ ] Wrote 3–4 sentences in your report describing what the EDA shows (e.g. class balance ratio, most correlated features with target)

---

## Feature Engineering & Model Development

`src/preprocessing.py` defines the reusable cleaning + feature pipeline (median/most-frequent imputation, scaling, one-hot encoding). `src/train.py` trains **two models** — Logistic Regression and Random Forest — each with a small hyperparameter grid, using 5-fold stratified cross-validation.

```bash
python src/train.py --data data/heart_disease_raw.csv
```

This prints cross-validated and held-out test metrics (accuracy, precision, recall, ROC-AUC) for both models and saves the better one to `models/heart_disease_model.joblib`.

**Checklist**
- [ ] Both models trained without errors
- [ ] Recorded accuracy/precision/recall/ROC-AUC for both models in your report
- [ ] One paragraph explaining *why* the winning model was chosen (e.g. higher ROC-AUC, better recall for a medical use-case where false negatives are costly)

---

## Experiment Tracking

MLflow logging is already wired into `src/train.py` (params, metrics, ROC curve plot, and the model itself as an artifact) — it runs automatically as part of the training step above.

View it:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

**Checklist**
- [ ] Opened the MLflow UI and confirmed 2 runs exist (one per model)
- [ ] Screenshot the run comparison table for your report
- [ ] Screenshot one run's artifacts tab showing the logged ROC curve + model

---

## Model Packaging & Reproducibility

Already handled:
- `models/heart_disease_model.joblib` — the fitted preprocessing pipeline **and** model bundled as one object (loading it is enough to predict on raw JSON — no separate transformer needed)
- `requirements.txt` — exact pinned versions of every dependency
- `src/preprocessing.py` — the transformer logic itself, reusable independently of the trained model

**Checklist**
- [ ] Confirmed `models/heart_disease_model.joblib` exists after training
- [ ] Confirmed `pip install -r requirements.txt` works from a **clean** virtualenv

---

## CI/CD Pipeline & Automated Testing

Unit tests live in `tests/` (8 tests: 4 for preprocessing, 4 for the API), and the pipeline is defined in `.github/workflows/ci.yml`. It:
1. Lints with `flake8`
2. Runs `pytest` and uploads the results as an artifact
3. Downloads the dataset and retrains the model
4. Builds the Docker image

Run locally before pushing:
```bash
pip install flake8
flake8 src api tests
pytest tests/ -v
```

**Checklist**
- [ ] Pushed this repo to **your own GitHub repo**
- [ ] Confirmed the Actions tab shows a green run (screenshot it)
- [ ] Deliberately broke a test once to confirm the pipeline **fails loudly** (screenshot the red X + logs), then revert

---

## Model Containerization

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest

# In another terminal:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Expected response shape: `{"prediction": 0 or 1, "label": "...", "confidence": 0.xx}`.

**Checklist**
- [ ] Docker image builds without errors
- [ ] Container runs and `/predict` returns a valid JSON response (screenshot the curl output)
- [ ] `/health` returns `{"status": "ok"}`

---

## Production Deployment

Manifests are in `k8s/`. With Minikube or Docker Desktop Kubernetes:

```bash
# Make the local image visible to the cluster (Minikube):
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get pods                      # confirm 2/2 pods Running
minikube service heart-disease-api-service --url   # get the access URL
```

Test the returned URL the same way as the curl command above.

**Checklist**
- [ ] `kubectl get pods` shows pods Running (screenshot)
- [ ] `kubectl get svc` shows the LoadBalancer/NodePort with an external URL (screenshot)
- [ ] Successful `/predict` call against the deployed URL (screenshot)

---

## Monitoring & Logging

- Every API request is logged to `api_requests.log` (see `api/main.py`) — input, prediction, and confidence, with timestamps.
- A Prometheus-compatible `/metrics` endpoint exposes `predict_requests_total` and `predict_latency_seconds`.
- `monitoring/prometheus.yml` is a ready-to-use scrape config.

```bash
docker run -p 9090:9090 -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
# open http://localhost:9090, query predict_requests_total
```

Optionally point Grafana at that Prometheus instance and build a 1-panel dashboard (request count over time).

**Checklist**
- [ ] `api_requests.log` shows entries after a few test predictions (screenshot or paste excerpt)
- [ ] `/metrics` endpoint reachable, shows non-zero `predict_requests_total` after some traffic
- [ ] (Bonus) Grafana dashboard screenshot

---

## Documentation & Reporting

Your final report (10 pages, doc/docx/pdf) should include:
- Setup/install instructions (can largely reuse this README)
- EDA and modelling choices
- Experiment tracking summary
- An architecture diagram (data → preprocessing → model → FastAPI → Docker → Kubernetes → monitoring) — a simple boxes-and-arrows diagram is enough
- CI/CD and deployment workflow screenshots
- Link to your GitHub repository

**This is the part you still need to personalize** — I can't generate your GitHub URL, your screenshots, or your demo video for you, but I'm happy to help you draft the report text itself, or build the architecture diagram, once you've got your repo set up and have run through the tasks above.

---

## Full Deliverables Checklist (from the assignment)

- [ ] GitHub repo containing: code, Dockerfile(s), requirements.txt, cleaned dataset + download script, notebooks/scripts, `tests/` folder, GitHub Actions YAML, deployment manifests, screenshots folder
- [ ] Final written report, 10 pages, doc/docx/pdf
- [ ] Deployed API URL or local access instructions
- [ ] Short video recording of the overall pipeline

## Repo layout

```
mlops-heart-disease/
├── data/download_data.py       # Data Acquisition
├── src/eda.py                  # EDA
├── src/preprocessing.py        # Feature Engineering, Packaging
├── src/train.py                # Model Development, Experiment Tracking, Packaging
├── api/main.py                 # Containerization, Monitoring
├── tests/                      # CI/CD
├── .github/workflows/ci.yml    # CI/CD
├── Dockerfile, .dockerignore   # Containerization
├── k8s/deployment.yaml, service.yaml   # Production Deployment
├── monitoring/prometheus.yml   # Monitoring
├── requirements.txt
└── README.md                   # this file
```
