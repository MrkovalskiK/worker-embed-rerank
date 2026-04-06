from pydantic_settings import BaseSettings


class WorkerConfig(BaseSettings):
    embedding_model: str
    rerank_model: str
    max_concurrency: int = 256
    max_batch_size: int = 0  # 0 = unlimited

    # GPU memory settings — tuned independently per model
    embedding_gpu_memory_utilization: float = 0.40
    embedding_max_num_seqs: int = 32
    embedding_max_model_len: int = 4096
    embedding_quantization: str | None = None

    rerank_gpu_memory_utilization: float = 0.40
    rerank_max_num_seqs: int = 32
    rerank_max_model_len: int = 4096
    rerank_quantization: str | None = None

    rerank_chat_template: str | None = None
