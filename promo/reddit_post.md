# Reddit Post Draft — r/MachineLearning
*[Draft — DO NOT PUBLISH without explicit approval]*

---

**Title:** I built a RAG system that auto-evaluates its own hallucination rate using the RAG Triad — here's the architecture [I built this]

---

**Body:**

The part I find most interesting about RAG isn't the retrieval — it's that hallucination is actually *measurable* in real time, and most production systems don't do it.

I built an end-to-end RAG system where every response is automatically scored before being returned:

**The RAG Triad:**
- **Faithfulness** (0.94): are the answer's claims derivable from the retrieved context?
- **Answer Relevance** (0.91): does the response address the actual query?
- **Context Precision** (0.88): what fraction of retrieved chunks were actually useful?

If any score falls below a threshold (configurable per metric), the system flags the response.

**Architecture:**
- `RecursiveCharacterSplitter` (chunk_size=500, overlap=50) → FAISS normalized cosine similarity search → zero-hallucination prompt template with enforced `[Source: file (chunk X)]` citation → `RAGEvaluator`
- FastAPI backend, `pyproject.toml` dependency management

**Numbers (internal evaluation, 30 QA pairs, not peer-reviewed):**
- Retrieval latency: 14ms (k=4, FAISS flat index)
- Faithfulness: 0.94 | Relevance: 0.91 | Precision: 0.88

**Runs without API key:** the `mock` LLM provider lets you test the full retrieval + evaluation pipeline deterministically.

**Source:** https://github.com/HiteshReddy2002/enterprise-rag-evaluation-engine

Honest question for the sub: how are people measuring faithfulness in production RAG? The heuristic approach I implemented (keyword overlap between answer sentences and context) is fast but obviously limited. NLI-based faithfulness scoring (like TruLens/RAGAS does) is more principled but adds latency. What's the practical tradeoff at scale?
