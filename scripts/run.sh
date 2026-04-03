#!/usr/bin/env bash
set -euo pipefail

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"

echo "EMBEDDING_MODEL=${EMBEDDING_MODEL}"
echo "RERANK_MODEL=${RERANK_MODEL}"

exec uv run python -m worker.main --rp_serve_api
