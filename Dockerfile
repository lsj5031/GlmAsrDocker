# Multi-stage build to minimize final image size
# Stage 1: Builder
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements-client.txt .

# Install dependencies into the venv.
# 1. Install PyTorch with specific index
# 2. Install requirements
# 3. Uninstall heavy, unused NVIDIA packages to save ~2GB+ in the builder cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt -r requirements-client.txt && \
    pip uninstall -y triton nvidia-cusolver-cu12 || true

# Final stage
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Only install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the optimized virtual environment from the builder stage
COPY --from=builder /app/venv $VIRTUAL_ENV
COPY --chown=appuser:appuser server.py glm_asr_cli.py .

# Create CLI entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "cli" ] || [ "$1" = "glm-asr" ]; then\n\
    shift\n\
    exec python /app/glm_asr_cli.py "$@"\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
