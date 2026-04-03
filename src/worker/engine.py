from __future__ import annotations

import threading
from abc import abstractmethod

from vllm import LLM

from .config import WorkerConfig


class BaseEngine:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._model: LLM | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> LLM:
        with self._lock:
            if self._model is None:
                self._model = self._create_model()
            return self._model

    @abstractmethod
    def _create_model(self) -> LLM:
        ...

    def warm_up(self) -> None:
        self._ensure_model()

    @abstractmethod
    def models(self) -> list[dict]:
        ...


class EmbeddingEngine(BaseEngine):
    def _create_model(self) -> LLM:
        return LLM(
            model=self.config.embedding_model,
            gpu_memory_utilization=self.config.embedding_gpu_memory_utilization,
            max_num_seqs=self.config.embedding_max_num_seqs,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        outputs = model.embed(texts)
        return [output.outputs.embedding for output in outputs]

    def models(self) -> list[dict]:
        return [{"id": self.config.embedding_model, "object": "model", "type": "embedding"}]


class ScoringEngine(BaseEngine):
    def _create_model(self) -> LLM:
        return LLM(
            model=self.config.rerank_model,
            runner="pooling",
            gpu_memory_utilization=self.config.rerank_gpu_memory_utilization,
            max_num_seqs=self.config.rerank_max_num_seqs,
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        model = self._ensure_model()
        score_kwargs = {}
        if self.config.rerank_chat_template is not None:
            score_kwargs["chat_template"] = self.config.rerank_chat_template
        outputs = model.score([query] * len(documents), documents, **score_kwargs)
        results = [
            {"index": i, "relevance_score": float(output.outputs.score)}
            for i, output in enumerate(outputs)
        ]
        if top_n is not None:
            results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
        return results

    def models(self) -> list[dict]:
        return [{"id": self.config.rerank_model, "object": "model", "type": "rerank"}]
