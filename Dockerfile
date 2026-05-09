# Python 3.11 slim — good balance of size and TF/transformers compatibility
FROM python:3.11-slim

# System deps needed by TensorFlow, SHAP (llvm), sentence-transformers, and pandas/sklearn builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace model cache into the image so cold starts don't have to hit the internet.
# Comment this block out if your nodes/ code uses different models — tell me which one and I'll adjust.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Copy the rest of the app
COPY . .

# Cloud Run injects $PORT (defaults to 8080). Streamlit must bind to 0.0.0.0.
ENV PORT=8080
EXPOSE 8080

# Streamlit needs headless mode and CORS off for Cloud Run's proxy
CMD streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
