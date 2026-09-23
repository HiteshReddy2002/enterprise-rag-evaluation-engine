"""RAG Generation Chain with context retrieval and synthesis."""

from typing import Dict, List, Optional
from src.ingestion.document_loader import Document
from src.vectorstore.vector_manager import VectorStore

try:
    from pydantic import BaseModel

    class RAGResponse(BaseModel):
        """Structured response from the RAG Chain."""
        query: str
        answer: str
        sources: List[Dict[str, str]]
        retrieved_contexts: List[str]
        confidence_score: float = 1.0

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class RAGResponse:
        query: str
        answer: str
        sources: List[Dict[str, str]]
        retrieved_contexts: List[str]
        confidence_score: float = 1.0




class RAGChain:
    """Orchestrates retrieval and grounded generation."""

    def __init__(self, vector_store: VectorStore, llm_provider: str = "mock"):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    def _format_context(self, documents: List[Document]) -> str:
        formatted_chunks = []
        for idx, doc in enumerate(documents, 1):
            source_tag = doc.metadata.get("filename", doc.id)
            chunk_num = doc.metadata.get("chunk_index", "0")
            formatted_chunks.append(f"[{idx}] (Source: {source_tag}, Chunk: {chunk_num})\n{doc.content}")
        return "\n\n".join(formatted_chunks)

    def _generate_answer(self, query: str, context: str, documents: List[Document]) -> str:
        if not documents:
            return "Based on the provided context, I cannot answer this question because no relevant documents were found."

        if self.llm_provider == "mock":
            # Deterministic, grounded mock generator for testing & demonstrations
            top_doc = documents[0]
            # Synthesize grounded answer
            sentences = [s.strip() for s in top_doc.content.split(".") if len(s.strip()) > 15]
            summary_snippet = sentences[0] if sentences else top_doc.content[:150]
            source_info = top_doc.metadata.get("filename", "document")
            chunk_idx = top_doc.metadata.get("chunk_index", "0")
            return f"{summary_snippet}. [Source: {source_info} (chunk {chunk_idx})]"
        else:
            # Fallback/external LLM invocation placeholder
            return f"Answer based on context: {documents[0].content[:200]}..."

    def query(self, query_text: str, top_k: int = 4) -> RAGResponse:
        retrieved_docs = self.vector_store.similarity_search(query_text, top_k=top_k)
        context_str = self._format_context(retrieved_docs)
        answer = self._generate_answer(query_text, context_str, retrieved_docs)

        sources = []
        for doc in retrieved_docs:
            sources.append({
                "id": doc.id,
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "0"),
                "score": f"{doc.score:.4f}" if doc.score is not None else "N/A"
            })

        return RAGResponse(
            query=query_text,
            answer=answer,
            sources=sources,
            retrieved_contexts=[d.content for d in retrieved_docs],
            confidence_score=float(retrieved_docs[0].score) if retrieved_docs and retrieved_docs[0].score else 0.85
        )
