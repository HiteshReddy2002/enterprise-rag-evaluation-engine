"""RAG Triad automated evaluation pipeline (Faithfulness, Answer Relevance, Context Precision)."""

import re
from typing import Dict, List
from src.chain.rag_chain import RAGResponse

try:
    from pydantic import BaseModel, Field

    class EvaluationMetric(BaseModel):
        name: str
        score: float = Field(ge=0.0, le=1.0)
        passed: bool
        details: str

    class RAGEvaluationReport(BaseModel):
        query: str
        faithfulness: EvaluationMetric
        answer_relevance: EvaluationMetric
        context_precision: EvaluationMetric
        overall_score: float
        passed_all: bool

except ImportError:
    from dataclasses import dataclass

    @dataclass
    class EvaluationMetric:
        name: str
        score: float
        passed: bool
        details: str

    @dataclass
    class RAGEvaluationReport:
        query: str
        faithfulness: EvaluationMetric
        answer_relevance: EvaluationMetric
        context_precision: EvaluationMetric
        overall_score: float
        passed_all: bool



class RAGEvaluator:
    """Evaluates RAG pipeline outputs against RAG Triad principles without hallucination."""

    def __init__(
        self,
        faithfulness_thresh: float = 0.80,
        relevance_thresh: float = 0.75,
        precision_thresh: float = 0.70,
    ):
        self.faithfulness_thresh = faithfulness_thresh
        self.relevance_thresh = relevance_thresh
        self.precision_thresh = precision_thresh

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> EvaluationMetric:
        """Measures whether claims in the answer can be directly inferred from the retrieved contexts."""
        if not answer or not contexts:
            return EvaluationMetric(name="Faithfulness", score=0.0, passed=False, details="Missing answer or context")

        combined_context = " ".join(contexts).lower()
        # Token overlap / groundedness check
        words_in_answer = set(re.findall(r"\w+", answer.lower()))
        # Filter out common stop words
        stopwords = {"the", "a", "an", "in", "on", "of", "and", "is", "to", "for", "with", "this", "that", "source", "chunk"}
        significant_words = words_in_answer - stopwords

        if not significant_words:
            return EvaluationMetric(name="Faithfulness", score=1.0, passed=True, details="No ungrounded claims")

        grounded_count = sum(1 for word in significant_words if word in combined_context)
        score = round(grounded_count / len(significant_words), 3)

        passed = score >= self.faithfulness_thresh
        return EvaluationMetric(
            name="Faithfulness",
            score=score,
            passed=passed,
            details=f"{grounded_count}/{len(significant_words)} key terms strictly grounded in context"
        )

    def evaluate_answer_relevance(self, query: str, answer: str) -> EvaluationMetric:
        """Measures whether the generated answer directly addresses the prompt."""
        if not query or not answer:
            return EvaluationMetric(name="Answer Relevance", score=0.0, passed=False, details="Empty query or answer")

        query_words = set(re.findall(r"\w+", query.lower())) - {"what", "how", "why", "is", "are", "the", "a", "an", "of", "in"}
        answer_words = set(re.findall(r"\w+", answer.lower()))

        if not query_words:
            score = 0.90
        else:
            overlap = sum(1 for w in query_words if w in answer_words)
            ratio = overlap / len(query_words)
            score = round(min(1.0, 0.6 + 0.4 * ratio), 3)

        passed = score >= self.relevance_thresh
        return EvaluationMetric(
            name="Answer Relevance",
            score=score,
            passed=passed,
            details=f"Answer aligns with user intent (relevance score: {score})"
        )


    def evaluate_context_precision(self, query: str, contexts: List[str]) -> EvaluationMetric:
        """Measures the signal-to-noise ratio of the retrieved contexts."""
        if not contexts:
            return EvaluationMetric(name="Context Precision", score=0.0, passed=False, details="No contexts retrieved")

        query_words = set(re.findall(r"\w+", query.lower())) - {"the", "a", "is", "in", "for", "to"}
        relevant_chunks = 0

        for ctx in contexts:
            ctx_lower = ctx.lower()
            if any(w in ctx_lower for w in query_words):
                relevant_chunks += 1

        score = round(relevant_chunks / len(contexts), 3)
        passed = score >= self.precision_thresh
        return EvaluationMetric(
            name="Context Precision",
            score=score,
            passed=passed,
            details=f"{relevant_chunks}/{len(contexts)} retrieved chunks contain query key terms"
        )

    def evaluate(self, response: RAGResponse) -> RAGEvaluationReport:
        faith = self.evaluate_faithfulness(response.answer, response.retrieved_contexts)
        relev = self.evaluate_answer_relevance(response.query, response.answer)
        prec = self.evaluate_context_precision(response.query, response.retrieved_contexts)

        overall = round((faith.score + relev.score + prec.score) / 3.0, 3)
        all_passed = faith.passed and relev.passed and prec.passed

        return RAGEvaluationReport(
            query=response.query,
            faithfulness=faith,
            answer_relevance=relev,
            context_precision=prec,
            overall_score=overall,
            passed_all=all_passed
        )
