# runpod-worker

RunPod serverless worker for text embedding and reranking using [vLLM](https://github.com/vllm-project/vllm).

## Routes

| Route | Format | Description |
|-------|--------|-------------|
| `POST /v1/embeddings` | OpenAI | Generate text embeddings |
| `POST /v1/rerank` | Cohere | Rerank documents by query relevance |
| `GET /v1/models` | OpenAI | List loaded models |

All routes use the RunPod serverless job input envelope:

```json
{
  "input": {
    "openai_route": "/v1/embeddings",
    "openai_input": { ... }
  }
}
```

## Configuration

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `EMBEDDING_MODEL` | Yes | — | vLLM embedding model name (HuggingFace) |
| `RERANK_MODEL` | Yes | — | vLLM reranker model name (HuggingFace) |
| `MAX_CONCURRENCY` | No | 300 | Max concurrent RunPod jobs |
| `GPU_MEMORY_UTILIZATION` | No | 0.45 | GPU memory fraction per model (two models share the GPU) |
| `MAX_NUM_SEQS` | No | 32 | Max sequences per vLLM engine |
| `RERANK_CHAT_TEMPLATE` | No | — | Jinja chat template string for reranker (uses model default if unset) |

Both models are loaded at startup. Model weights are downloaded from HuggingFace and cached to `~/.cache/huggingface`.

## Local Development

```bash
./scripts/run.sh
```

Override models via env vars:

```bash
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B ./scripts/run.sh
```

The worker starts on port 8000. Both models load sequentially at startup before requests are accepted.

### Test embed

```bash
curl -X POST http://localhost:8000 \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "openai_route": "/v1/embeddings",
      "openai_input": {
        "input": ["hello world"],
        "model": "Qwen/Qwen3-Embedding-0.6B"
      }
    }
  }'
```

### Test rerank

```bash
curl -X POST http://localhost:8000 \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "openai_route": "/v1/rerank",
      "openai_input": {
        "query": "what is machine learning?",
        "documents": [
          "Machine learning is a subset of AI.",
          "Paris is the capital of France."
        ],
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "top_n": 2
      }
    }
  }'
```

### Test models

```bash
curl -X POST http://localhost:8000 \
  -H 'Content-Type: application/json' \
  -d '{"input": {"openai_route": "/v1/models", "openai_input": {}}}'
```

## Deploy to RunPod

1. Build and push the image to a registry
2. Create a RunPod serverless endpoint using the image
3. Set `EMBEDDING_MODEL` and `RERANK_MODEL` environment variables on the endpoint

## Project Structure

```
src/worker/
  main.py      # Entry point — eager model loading + RunPod start
  handler.py   # Route dispatch factory (make_handler)
  engine.py    # EmbeddingEngine + ScoringEngine (vLLM)
  config.py    # Pydantic settings from env vars
  utils.py     # Logging setup
scripts/
  start.sh     # Container entrypoint
  run.sh       # Local dev runner
Dockerfile     # nvidia/cuda:12.1.0-runtime-ubuntu22.04 + uv
```
