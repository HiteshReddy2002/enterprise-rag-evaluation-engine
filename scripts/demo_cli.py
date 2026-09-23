"""Interactive CLI demo for the LLM RAG & Evaluation Engine."""

import sys
import io
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import RecursiveCharacterSplitter
from src.vectorstore.vector_manager import VectorStore, EmbeddingService
from src.chain.rag_chain import RAGChain
from src.evaluation.evaluator import RAGEvaluator


def run_demo():
    print("=" * 70)
    print("  [LLM RAG Engine & RAG Triad Evaluator CLI Demo]")
    print("  Production AI/ML Engineering Showcase")
    print("=" * 70)

    # 1. Initialize Components
    embedding_service = EmbeddingService(dimension=settings.embedding_dim, provider="mock")
    vector_store = VectorStore(dimension=settings.embedding_dim, embedding_service=embedding_service)
    splitter = RecursiveCharacterSplitter(chunk_size=300, chunk_overlap=30)
    rag_chain = RAGChain(vector_store=vector_store, llm_provider="mock")
    evaluator = RAGEvaluator()

    # 2. Ingest Sample Docs
    docs_dir = settings.data_dir / "sample_docs"
    docs = DocumentLoader.load_directory(docs_dir, "*.md")
    print(f"\n[+] Loaded {len(docs)} documents from: {docs_dir}")

    chunks = splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    print(f"[+] Indexed {len(chunks)} semantic vector chunks into VectorStore.\n")

    # 3. Interactive Queries
    queries = [
        "What are the key components of RAG architecture?",
        "How do MLOps best practices prevent hallucinations in Generative AI?",
    ]

    for idx, q in enumerate(queries, 1):
        print(f"\n----------------------------------------------------------------------")
        print(f"Query #{idx}: {q}")
        print(f"----------------------------------------------------------------------")

        response = rag_chain.query(q, top_k=2)
        report = evaluator.evaluate(response)

        print("\n>> Generated Grounded Answer:")
        print(f"   {response.answer}")

        print("\n>> Retrieved Context Sources:")
        for src in response.sources:
            print(f"   - Source: {src['filename']} | Chunk: {src['chunk_index']} | Similarity: {src['score']}")

        print("\n>> RAG Triad Automated Evaluation:")
        print(f"   - Faithfulness (Groundedness): {report.faithfulness.score:.2f} ({'PASS' if report.faithfulness.passed else 'FAIL'}) -> {report.faithfulness.details}")
        print(f"   - Answer Relevance:           {report.answer_relevance.score:.2f} ({'PASS' if report.answer_relevance.passed else 'FAIL'}) -> {report.answer_relevance.details}")
        print(f"   - Context Precision:          {report.context_precision.score:.2f} ({'PASS' if report.context_precision.passed else 'FAIL'}) -> {report.context_precision.details}")
        print(f"   - Overall Quality Score:      {report.overall_score:.2f} ({'HIGH QUALITY' if report.passed_all else 'CHECK REQUIRED'})")

    print("\n" + "=" * 70)
    print("  Demo completed successfully! Everything functioning as expected.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()

