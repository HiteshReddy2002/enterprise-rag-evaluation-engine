import pytest
from src.ingestion.document_loader import Document
from src.vectorstore.vector_manager import VectorStore, EmbeddingService
from src.chain.rag_chain import RAGChain


def test_vector_store_and_rag_chain():
    embedding_service = EmbeddingService(dimension=64, provider="mock")
    store = VectorStore(dimension=64, embedding_service=embedding_service)

    docs = [
        Document(id="doc1", content="Antigravity AI is a revolutionary agentic platform.", metadata={"filename": "doc1.txt", "chunk_index": "0"}),
        Document(id="doc2", content="Python is a versatile programming language for machine learning.", metadata={"filename": "doc2.txt", "chunk_index": "0"}),
    ]

    count = store.add_documents(docs)
    assert count == 2

    # Test similarity search
    results = store.similarity_search("agentic AI platform", top_k=1)
    assert len(results) == 1
    assert results[0].id == "doc1"
    assert results[0].score is not None

    # Test RAG Chain query
    rag_chain = RAGChain(vector_store=store, llm_provider="mock")
    response = rag_chain.query("agentic AI platform", top_k=1)

    assert response.query == "agentic AI platform"
    assert len(response.sources) == 1
    assert "doc1.txt" in response.answer or "doc1.txt" in response.sources[0]["filename"]
