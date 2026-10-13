# 1. Force the image to always be built as amd64
FROM --platform=linux/amd64 python:3.11-slim

# System deps needed by TensorFlow, SHAP, sentence-transformers, and pandas/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Disable experimental TensorFlow optimizations that can cause segfaults
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install Python deps first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# COMMENTED OUT TO PREVENT CLOUD BUILD TIMEOUT (downloads will happen at runtime instead)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Copy the rest of the app
COPY . .

# Cloud Run injects $PORT (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Streamlit needs headless mode and CORS off
CMD exec streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
