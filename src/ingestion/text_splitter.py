"""Text splitting and semantic chunking mechanisms."""

import re
from typing import List
from src.ingestion.document_loader import Document


class RecursiveCharacterSplitter:
    """Splits text into chunks maintaining paragraph and sentence boundary coherence."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks: List[str] = []
        current_chunk = ""

        # Simple semantic-aware paragraph splitting
        paragraphs = re.split(r"(\n\n|\n)", text)
        
        for piece in paragraphs:
            if not piece.strip():
                continue

            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + piece.strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep overlap from the end of current_chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:].strip() + " " + piece.strip()
                else:
                    # Piece is longer than chunk_size, hard slice
                    for i in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                        chunks.append(piece[i : i + self.chunk_size].strip())

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            raw_chunks = self.split_text(doc.content)
            for idx, chunk in enumerate(raw_chunks):
                chunk_id = f"{doc.id}_chunk_{idx}"
                metadata = {
                    **doc.metadata,
                    "chunk_index": str(idx),
                    "total_chunks": str(len(raw_chunks)),
                    "parent_id": doc.id,
                }
                chunked_docs.append(Document(id=chunk_id, content=chunk, metadata=metadata))
        return chunked_docs
