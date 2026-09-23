"""Vector store manager and embedding utilities."""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.ingestion.document_loader import Document



class EmbeddingService:
    """Generates dense vector embeddings. Uses keyword/token-aware hashed representation for offline zero-cost testing and API embeddings in production."""

    def __init__(self, dimension: int = 384, provider: str = "mock"):
        self.dimension = dimension
        self.provider = provider

    def embed_text(self, text: str) -> np.ndarray:
        """Deterministically embed text into a normalized dense vector with semantic token hashing."""
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        if self.provider == "mock":
            vec = np.zeros(self.dimension, dtype=np.float32)
            # Tokenize into word tokens
            tokens = re.findall(r"\w+", text.lower())
            for token in tokens:
                # Hash token to a dimension index
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
                vec[idx] += sign * (1.0 + (1.0 / (len(token) + 1)))

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        else:
            # Production external embedding API placeholder
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.normal(loc=0.0, scale=1.0, size=self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)


    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)


class VectorStore:
    """In-memory & persisted Vector Store with Cosine Similarity Search."""

    def __init__(self, dimension: int = 384, embedding_service: EmbeddingService = None):
        self.dimension = dimension
        self.embedding_service = embedding_service or EmbeddingService(dimension=dimension)
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_documents(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        texts = [d.content for d in docs]
        new_vectors = self.embedding_service.embed_batch(texts)
        
        if len(self.documents) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.documents.extend(docs)
        return len(docs)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Document]:
        if len(self.documents) == 0:
            return []

        query_vec = self.embedding_service.embed_text(query).reshape(1, -1)
        
        # Cosine similarity for normalized vectors = dot product
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Rank by score descending
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx].model_copy(deep=True)
            doc.score = float(scores[idx])
            results.append(doc)

        return results

    def save_to_disk(self, directory: Path | str) -> None:
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metadata / docs
        docs_data = [d.model_dump() for d in self.documents]
        with open(save_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        # Save vectors
        np.save(save_path / "vectors.npy", self.vectors)

    def load_from_disk(self, directory: Path | str) -> bool:
        load_path = Path(directory)
        docs_file = load_path / "documents.json"
        vecs_file = load_path / "vectors.npy"

        if not docs_file.exists() or not vecs_file.exists():
            return False

        with open(docs_file, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            self.documents = [Document(**d) for d in docs_data]

        self.vectors = np.load(vecs_file)
        return True
