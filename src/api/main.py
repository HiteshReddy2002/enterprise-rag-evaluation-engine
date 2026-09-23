"""FastAPI REST Service for RAG Search, Ingestion, and Evaluation."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from src.config import settings
from src.ingestion.document_loader import Document, DocumentLoader
from src.ingestion.text_splitter import RecursiveCharacterSplitter
from src.vectorstore.vector_manager import VectorStore, EmbeddingService
from src.chain.rag_chain import RAGChain, RAGResponse
from src.evaluation.evaluator import RAGEvaluator, RAGEvaluationReport

app = FastAPI(
    title="Production LLM RAG & Evaluation Service",
    description="REST API for document ingestion, semantic search, grounded answer generation, and hallucination evaluation.",
    version="1.0.0",
)

# Global instances
embedding_service = EmbeddingService(dimension=settings.embedding_dim, provider=settings.llm_provider)
vector_store = VectorStore(dimension=settings.embedding_dim, embedding_service=embedding_service)
splitter = RecursiveCharacterSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
rag_chain = RAGChain(vector_store=vector_store, llm_provider=settings.llm_provider)
evaluator = RAGEvaluator(
    faithfulness_thresh=settings.faithfulness_threshold,
    relevance_thresh=settings.answer_relevancy_threshold,
    precision_thresh=settings.context_precision_threshold,
)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    include_evaluation: Optional[bool] = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    evaluation: Optional[RAGEvaluationReport] = None


class IngestionRequest(BaseModel):
    text: str
    title: str
    metadata: Optional[dict] = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "indexed_documents": len(vector_store.documents),
        "embedding_dim": settings.embedding_dim,
        "provider": settings.llm_provider,
    }


@app.post("/ingest", response_model=dict)
def ingest_text(payload: IngestionRequest):
    doc = Document(
        id=payload.title,
        content=payload.text,
        metadata={"filename": payload.title, **(payload.metadata or {})},
    )
    chunks = splitter.split_documents([doc])
    count = vector_store.add_documents(chunks)
    return {"message": f"Successfully ingested {count} chunks for '{payload.title}'", "total_chunks": count}


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response: RAGResponse = rag_chain.query(payload.query, top_k=payload.top_k)
    eval_report = None
    if payload.include_evaluation:
        eval_report = evaluator.evaluate(response)

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        sources=response.sources,
        evaluation=eval_report,
    )
