FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact
COPY src/ src/
COPY api/ api/
COPY models/ models/

ENV MODEL_PATH=models/heart_disease_model.joblib
EXPOSE 8000

# Basic container-level healthcheck used by Kubernetes / docker run --health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
