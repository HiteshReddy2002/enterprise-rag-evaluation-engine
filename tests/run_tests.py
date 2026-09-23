"""Pytest and standard unittest test runner script."""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document_loader import Document, DocumentLoader

from src.ingestion.text_splitter import RecursiveCharacterSplitter
from src.vectorstore.vector_manager import VectorStore, EmbeddingService
from src.chain.rag_chain import RAGChain, RAGResponse
from src.evaluation.evaluator import RAGEvaluator


class TestIngestion(unittest.TestCase):
    def test_document_loader(self):
        doc = Document(id="sample.txt", content="Hello RAG world!", metadata={"filename": "sample.txt"})
        self.assertEqual(doc.id, "sample.txt")
        self.assertIn("Hello RAG", doc.content)

    def test_splitter(self):
        text = "Section 1 content with details.\n\nSection 2 has more relevant information.\n\nSection 3 is the conclusion."
        doc = Document(id="doc1", content=text, metadata={"filename": "doc1.md"})
        splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=10)
        chunks = splitter.split_documents([doc])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["parent_id"], "doc1")


class TestRAGChainAndVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding_service = EmbeddingService(dimension=64, provider="mock")
        self.store = VectorStore(dimension=64, embedding_service=self.embedding_service)

    def test_similarity_search_and_query(self):
        docs = [
            Document(id="doc1", content="Antigravity AI is an agentic platform.", metadata={"filename": "doc1.txt", "chunk_index": "0"}),
            Document(id="doc2", content="Python is a programming language for data science.", metadata={"filename": "doc2.txt", "chunk_index": "0"}),
        ]
        self.store.add_documents(docs)
        results = self.store.similarity_search("agentic platform", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "doc1")

        chain = RAGChain(vector_store=self.store, llm_provider="mock")
        response = chain.query("agentic platform", top_k=1)
        self.assertEqual(response.query, "agentic platform")
        self.assertGreater(len(response.sources), 0)


class TestRAGEvaluation(unittest.TestCase):
    def test_rag_triad(self):
        evaluator = RAGEvaluator(faithfulness_thresh=0.6, relevance_thresh=0.6, precision_thresh=0.6)
        query = "What is RAG?"
        context = "Retrieval augmented generation optimizes LLM outputs by fetching authoritative facts."
        answer = "Retrieval augmented generation optimizes LLM outputs with authoritative facts."
        
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[{"id": "doc1", "filename": "rag.md", "chunk_index": "0", "score": "0.95"}],
            retrieved_contexts=[context],
            confidence_score=0.95,
        )
        report = evaluator.evaluate(response)
        self.assertGreater(report.faithfulness.score, 0.5)
        self.assertGreater(report.answer_relevance.score, 0.5)
        self.assertGreater(report.overall_score, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
