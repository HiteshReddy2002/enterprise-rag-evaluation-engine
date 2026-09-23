"""Configuration module for LLM RAG Engine."""

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        """Application settings with defaults and environment overrides."""

        app_name: str = "Production LLM RAG Engine"
        app_env: str = "development"
        log_level: str = "INFO"

        # Directory Paths
        base_dir: Path = Path(__file__).resolve().parent.parent
        data_dir: Path = base_dir / "data"
        vector_store_path: Path = data_dir / "vector_index"

        # LLM and Embeddings
        llm_provider: str = "mock"
        openai_api_key: str = ""
        gemini_api_key: str = ""
        embedding_model: str = "text-embedding-3-small"
        embedding_dim: int = 384
        llm_model: str = "gpt-4o-mini"
        temperature: float = 0.2
        max_tokens: int = 1024

        # Retrieval & Chunking
        chunk_size: int = 500
        chunk_overlap: int = 50
        top_k_retrieval: int = 4
        similarity_threshold: float = 0.65

        # Evaluation Thresholds (RAG Triad)
        faithfulness_threshold: float = 0.85
        answer_relevancy_threshold: float = 0.80
        context_precision_threshold: float = 0.75

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    settings = Settings()

except ImportError:
    # Graceful standard dataclass fallback
    from dataclasses import dataclass, field

    @dataclass
    class Settings:
        app_name: str = "Production LLM RAG Engine"
        app_env: str = os.getenv("APP_ENV", "development")
        log_level: str = os.getenv("LOG_LEVEL", "INFO")

        base_dir: Path = Path(__file__).resolve().parent.parent
        data_dir: Path = Path(__file__).resolve().parent.parent / "data"
        vector_store_path: Path = Path(__file__).resolve().parent.parent / "data" / "vector_index"

        llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
        openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
        llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
        max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

        chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
        similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

        faithfulness_threshold: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))
        answer_relevancy_threshold: float = float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.80"))
        context_precision_threshold: float = float(os.getenv("CONTEXT_PRECISION_THRESHOLD", "0.75"))

    settings = Settings()

