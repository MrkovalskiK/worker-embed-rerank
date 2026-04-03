FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --no-dev --frozen

# Model cache dir — override at runtime with a network volume path (e.g. /runpod-volume/models)
ARG BASE_PATH="/runpod-volume"

ENV BASE_PATH=$BASE_PATH \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

# Optional: bake models into the image at build time
# docker build --build-arg EMBEDDING_MODEL=BAAI/bge-m3 --build-arg RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B .
ARG EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
ENV EMBEDDING_MODEL=$EMBEDDING_MODEL

ARG RERANK_MODEL="Qwen/Qwen3-Reranker-4B"
ENV RERANK_MODEL=$RERANK_MODEL

RUN mkdir -p "$HF_HOME" && \
    if [ -n "$EMBEDDING_MODEL" ]; then uv run python -m huggingface_hub download "$EMBEDDING_MODEL"; fi && \
    if [ -n "$RERANK_MODEL" ]; then uv run python -m huggingface_hub download "$RERANK_MODEL"; fi

# Copy source and entrypoint
COPY src/ ./src/
COPY scripts/start.sh ./scripts/start.sh
RUN chmod +x /app/scripts/start.sh

ENV PYTHONPATH=/app/src

CMD ["/app/scripts/start.sh"]
