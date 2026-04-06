FROM nvidia/cuda:12.8.1-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.8/compat/

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.19.0" --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

# Install app dependencies (after vLLM to avoid PyTorch version conflicts)
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install .

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

# Copy source and entrypoint
COPY src/worker/download_model.py ./src/worker/download_model.py

RUN mkdir -p "$HF_HOME" && \
    if [ -n "$EMBEDDING_MODEL" ]; then python3 /app/src/worker/download_model.py --name "$EMBEDDING_MODEL"; fi && \
    if [ -n "$RERANK_MODEL" ]; then python3 /app/src/worker/download_model.py --name "$RERANK_MODEL"; fi

COPY src/ ./src/

ENV PYTHONPATH=/app/src

ENV DEV=""

CMD sh -c "python3 -m worker.main ${DEV:+--rp_serve_api}"
