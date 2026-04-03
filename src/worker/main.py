from __future__ import annotations

import logging

import runpod

from .config import WorkerConfig
from .engine import EmbeddingEngine, ScoringEngine
from .handler import make_handler

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = WorkerConfig()
    embedding_engine = EmbeddingEngine(config)
    scoring_engine = ScoringEngine(config)

    # Eager-load both models sequentially before accepting requests.
    # Avoids deadlock from concurrent vLLM LLM initialization mid-request.
    embedding_engine.warm_up()
    scoring_engine.warm_up()

    handler = make_handler(embedding_engine, scoring_engine)

    def concurrency_modifier(current_concurrency: int) -> int:
        return min(current_concurrency, config.max_concurrency)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
