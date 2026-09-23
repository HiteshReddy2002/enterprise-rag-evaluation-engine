"""Data models and ingestion components for the RAG engine."""

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    class Document(BaseModel):
        """Represents an ingested document or chunk with metadata."""
        id: str
        content: str
        metadata: Dict[str, str] = Field(default_factory=dict)
        score: Optional[float] = None

        def model_copy(self, deep: bool = True):
            return Document(
                id=self.id,
                content=self.content,
                metadata=dict(self.metadata),
                score=self.score
            )

        def model_dump(self):
            return {
                "id": self.id,
                "content": self.content,
                "metadata": self.metadata,
                "score": self.score
            }

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        id: str
        content: str
        metadata: Dict[str, str] = field(default_factory=dict)
        score: Optional[float] = None

        def model_copy(self, deep: bool = True):
            return Document(
                id=self.id,
                content=self.content,
                metadata=dict(self.metadata),
                score=self.score
            )

        def model_dump(self):
            return {
                "id": self.id,
                "content": self.content,
                "metadata": self.metadata,
                "score": self.score
            }



class DocumentLoader:
    """Multi-format document loader supporting text, markdown, and unstructured formats."""

    @staticmethod
    def load_file(file_path: str | Path) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".txt", ".md", ".json", ".py", ".csv"]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            # Fallback for basic text read
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        return Document(
            id=path.name,
            content=content,
            metadata={"source": str(path), "filename": path.name, "type": suffix[1:] or "text"},
        )

    @staticmethod
    def load_directory(dir_path: str | Path, glob_pattern: str = "*.*") -> List[Document]:
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            return []

        documents = []
        for file_path in directory.glob(glob_pattern):
            if file_path.is_file():
                try:
                    doc = DocumentLoader.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        return documents
