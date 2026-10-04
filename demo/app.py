"""
Gradio demo for the Enterprise RAG Evaluation Engine.

Imports from the repo's actual src modules:
  - src.ingestion.document_loader     → Document, DocumentLoader
  - src.ingestion.text_splitter       → RecursiveCharacterSplitter
  - src.vectorstore.vector_manager    → VectorStore, EmbeddingService
  - src.chain.rag_chain               → RAGChain
  - src.evaluation.evaluator          → RAGEvaluator
  - src.config                        → settings

Run from the repo root:
    python -m gradio demo/app.py
    # or:
    python demo/app.py
"""

import os
import sys
import pathlib
import tempfile

import gradio as gr

# ── Allow imports from repo root ────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Import real modules ──────────────────────────────────────────────────────
try:
    from src.config import settings
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.text_splitter import RecursiveCharacterSplitter
    from src.vectorstore.vector_manager import VectorStore, EmbeddingService
    from src.chain.rag_chain import RAGChain
    from src.evaluation.evaluator import RAGEvaluator
    REAL_MODULES = True
except ImportError as e:
    print(f"[WARN] Could not import live modules: {e}. Using mock mode.")
    REAL_MODULES = False

# ── Global RAG system (initialized on first use) ────────────────────────────
_rag_system = {}


def _init_rag():
    """Lazily initialize the RAG pipeline."""
    if "chain" in _rag_system:
        return True, None
    if not REAL_MODULES:
        return False, "Live modules not available. Running in mock mode."
    try:
        embedding_service = EmbeddingService(
            dimension=settings.embedding_dim,
            provider=settings.llm_provider
        )
        vector_store = VectorStore(
            dimension=settings.embedding_dim,
            embedding_service=embedding_service
        )
        splitter = RecursiveCharacterSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chain = RAGChain(vector_store=vector_store, llm_provider=settings.llm_provider)
        evaluator = RAGEvaluator(
            faithfulness_thresh=settings.faithfulness_threshold,
            relevance_thresh=settings.answer_relevancy_threshold,
            precision_thresh=settings.context_precision_threshold,
        )
        _rag_system.update({
            "vector_store": vector_store,
            "splitter": splitter,
            "chain": chain,
            "evaluator": evaluator,
            "loader": DocumentLoader(),
        })
        return True, None
    except Exception as e:
        return False, str(e)


def ingest_text(text_content: str, doc_name: str = "demo_doc"):
    """Ingest raw text into the vector store."""
    ok, err = _init_rag()
    if not ok:
        return f"Mock mode: would ingest '{doc_name}' ({len(text_content)} chars). {err or ''}"

    try:
        from src.ingestion.document_loader import Document
        doc = Document(id=doc_name, content=text_content, metadata={"filename": doc_name})
        chunks = _rag_system["splitter"].split([doc])
        _rag_system["vector_store"].add_documents(chunks)
        return f"✅ Ingested '{doc_name}': {len(chunks)} chunks added to vector store."
    except Exception as e:
        return f"Error during ingestion: {e}"


def query_rag(question: str, run_eval: bool = True):
    """Query the RAG system and optionally run RAG Triad evaluation."""
    if not question.strip():
        return "Please enter a question.", "", ""

    ok, err = _init_rag()

    if not ok or not REAL_MODULES:
        # Mock response
        mock_answer = (
            f"[MOCK] Answer for: '{question}'\n\n"
            "This is a demonstration response. The actual RAG system retrieves "
            "semantically relevant document chunks and generates grounded answers "
            "with source citations.\n\n"
            "Set your LLM_PROVIDER and API key in .env to enable live responses."
        )
        mock_eval = "Faithfulness: N/A | Relevance: N/A | Precision: N/A (mock mode)"
        mock_sources = "No real documents indexed — add text in the 'Ingest' tab first."
        return mock_answer, mock_eval, mock_sources

    try:
        response = _rag_system["chain"].query(question)
        answer = response.answer
        sources_str = "\n".join(
            f"• [{s['filename']}] chunk {s['chunk_index']} (score: {s['score']})"
            for s in response.sources
        ) or "No sources retrieved."

        eval_str = ""
        if run_eval:
            try:
                report = _rag_system["evaluator"].evaluate(
                    query=question,
                    answer=answer,
                    contexts=response.retrieved_contexts,
                )
                eval_str = (
                    f"Faithfulness: {report.faithfulness_score:.2f} {'✅' if report.faithfulness_passed else '❌'} | "
                    f"Relevance: {report.answer_relevance_score:.2f} {'✅' if report.relevance_passed else '❌'} | "
                    f"Precision: {report.context_precision_score:.2f} {'✅' if report.precision_passed else '❌'}"
                )
            except Exception as e:
                eval_str = f"Evaluation error: {e}"

        return answer, eval_str, sources_str
    except Exception as e:
        return f"Query error: {e}", "", ""


# ── Gradio UI ────────────────────────────────────────────────────────────────

MODE = "🟢 LIVE" if REAL_MODULES else "🔴 MOCK MODE"

with gr.Blocks(title="Enterprise RAG Evaluation Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🧠 Enterprise RAG Evaluation Engine
    **Demo** | {MODE} | RAG Triad: Faithfulness · Answer Relevance · Context Precision

    > **Source**: [HiteshReddy2002/enterprise-rag-evaluation-engine](https://github.com/HiteshReddy2002/enterprise-rag-evaluation-engine)
    """)

    with gr.Tabs():
        with gr.Tab("📥 Ingest Documents"):
            gr.Markdown("Paste document text to add it to the vector store.")
            doc_name = gr.Textbox(label="Document Name", value="my_document", max_lines=1)
            doc_text = gr.Textbox(
                label="Document Content (paste text here)",
                lines=10,
                placeholder="Paste any text content here — articles, documentation, research notes...",
                value=(
                    "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
                    "information retrieval with language model generation. Instead of relying "
                    "solely on parametric knowledge, RAG retrieves relevant documents from a "
                    "knowledge base and grounds the LLM response in verified context. "
                    "This dramatically reduces hallucination rates and enables dynamic, "
                    "up-to-date knowledge without expensive fine-tuning."
                ),
            )
            ingest_btn = gr.Button("📥 Ingest", variant="primary")
            ingest_status = gr.Textbox(label="Status", interactive=False)
            ingest_btn.click(fn=ingest_text, inputs=[doc_text, doc_name], outputs=[ingest_status])

        with gr.Tab("🔍 Query & Evaluate"):
            gr.Markdown("Ask a question. The RAG system retrieves relevant context and evaluates its answer.")
            question_input = gr.Textbox(
                label="Question",
                placeholder="What is RAG and how does it reduce hallucinations?",
                lines=2,
                value="What is RAG and how does it reduce hallucinations?",
            )
            run_eval_cb = gr.Checkbox(label="Run RAG Triad Evaluation", value=True)
            query_btn = gr.Button("🔍 Ask", variant="primary")

            with gr.Row():
                answer_out = gr.Textbox(label="Answer", lines=8, interactive=False)
            with gr.Row():
                eval_out = gr.Textbox(label="RAG Triad Scores", interactive=False)
                sources_out = gr.Textbox(label="Retrieved Sources", lines=4, interactive=False)

            query_btn.click(
                fn=query_rag,
                inputs=[question_input, run_eval_cb],
                outputs=[answer_out, eval_out, sources_out],
            )

            gr.Examples(
                examples=[
                    ["What is RAG and how does it reduce hallucinations?", True],
                    ["What is the difference between RAG and fine-tuning?", True],
                    ["How does semantic chunking work?", True],
                ],
                inputs=[question_input, run_eval_cb],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
