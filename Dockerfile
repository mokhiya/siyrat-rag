FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.8 GB vs the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('intfloat/multilingual-e5-base')"

COPY app.py .
COPY src/ src/
COPY evaluation/ evaluation/
COPY results/ results/
COPY assets/ assets/
COPY .streamlit/ .streamlit/

COPY data/chroma/ data/chroma/
COPY data/processed/ data/processed/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:${PORT:-7860}/_stcore/health || exit 1

# Railway injects a PORT env variable; HF Spaces expects 7860 by default
CMD ["sh", "-c", "streamlit run app.py \
     --server.port=${PORT:-7860} \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --browser.gatherUsageStats=false"]
