import pytest
from src.chain.rag_chain import RAGResponse
from src.evaluation.evaluator import RAGEvaluator


def test_rag_triad_evaluation():
    evaluator = RAGEvaluator(
        faithfulness_thresh=0.70,
        relevance_thresh=0.70,
        precision_thresh=0.70,
    )

    query = "What is retrieval augmented generation?"
    context = "Retrieval augmented generation optimizes LLM outputs by fetching authoritative facts from external vector stores."
    answer = "Retrieval augmented generation optimizes LLM outputs with external vector store facts."

    response = RAGResponse(
        query=query,
        answer=answer,
        sources=[{"id": "doc1", "filename": "rag.md", "chunk_index": "0", "score": "0.95"}],
        retrieved_contexts=[context],
        confidence_score=0.95,
    )

    report = evaluator.evaluate(response)

    assert report.query == query
    assert report.faithfulness.score > 0.6
    assert report.answer_relevance.score > 0.6
    assert report.context_precision.score > 0.6
    assert report.overall_score > 0.6
