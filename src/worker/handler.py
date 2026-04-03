from __future__ import annotations

import sys

import torch

from .engine import EmbeddingEngine, ScoringEngine


def make_handler(embedding_engine: EmbeddingEngine, scoring_engine: ScoringEngine):
    max_batch_size = embedding_engine.config.max_batch_size

    async def handler(job: dict):
        try:
            job_input = job["input"]
            route = job_input.get("openai_route", "")
            body = job_input.get("openai_input", {})

            if route == "/v1/embeddings":
                texts = body.get("input", [])
                if isinstance(texts, str):
                    texts = [texts]
                if not texts:
                    yield {"object": "list", "data": [], "model": body.get("model", embedding_engine.config.embedding_model)}
                    return
                if max_batch_size and len(texts) > max_batch_size:
                    yield {"error": f"Batch size {len(texts)} exceeds maximum {max_batch_size}"}
                    return
                vectors = embedding_engine.embed(texts)
                yield {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": v}
                        for i, v in enumerate(vectors)
                    ],
                    "model": body.get("model", embedding_engine.config.embedding_model),
                }

            elif route == "/v1/rerank":
                query = body.get("query")
                documents = body.get("documents")
                if query is None or documents is None:
                    yield {"error": "Missing required fields: 'query' and 'documents'"}
                    return
                if not documents:
                    yield {"results": [], "model": body.get("model", scoring_engine.config.rerank_model)}
                    return
                if max_batch_size and len(documents) > max_batch_size:
                    yield {"error": f"Batch size {len(documents)} exceeds maximum {max_batch_size}"}
                    return
                results = scoring_engine.rerank(
                    query=query,
                    documents=documents,
                    top_n=body.get("top_n"),
                )
                yield {
                    "results": results,
                    "model": body.get("model", scoring_engine.config.rerank_model),
                }

            elif route == "/v1/models":
                yield {"object": "list", "data": embedding_engine.models() + scoring_engine.models()}

            else:
                yield {"error": f"Unknown route: {route!r}. Supported: /v1/embeddings, /v1/rerank, /v1/models"}

        except Exception as e:
            if isinstance(e, torch.cuda.CudaError) or (
                isinstance(e, RuntimeError) and ("CUDA" in str(e) or "out of memory" in str(e).lower())
            ):
                sys.exit(1)
            yield {"error": str(e)}

    return handler
