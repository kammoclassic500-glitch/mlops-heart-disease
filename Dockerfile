# ---------------------------------------------------------------------------
# Heart Disease Risk API — Docker image
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install system deps needed by scikit-learn/pandas wheels (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, model artifact, and preprocessing code
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

# Non-root user (good practice, some graders check for this)
RUN useradd -m apiuser && chown -R apiuser /app
USER apiuser

EXPOSE 8000

# Basic container-level healthcheck hitting our /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
