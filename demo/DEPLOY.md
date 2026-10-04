# Deploy RAG Evaluation Engine Demo to Hugging Face Spaces

This guide deploys the Enterprise RAG Evaluation Engine demo (Gradio) to [Hugging Face Spaces](https://huggingface.co/spaces).

## Prerequisites

- [Hugging Face account](https://huggingface.co/join)
- `huggingface_hub` CLI: `pip install huggingface_hub`
- An LLM API key (OpenAI or Gemini) — or deploy in mock mode (no key needed)

---

## Steps

### 1. Create a new Gradio Space

```bash
huggingface-cli login
huggingface-cli repo create rag-evaluation-demo --type space --space_sdk gradio
```

### 2. Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<YOUR_HF_USERNAME>/rag-evaluation-demo
cd rag-evaluation-demo
```

### 3. Copy demo + source files

```bash
# From the enterprise-rag-evaluation-engine repo root:
cp demo/app.py ../rag-evaluation-demo/app.py
cp demo/requirements-demo.txt ../rag-evaluation-demo/requirements.txt
cp -r src ../rag-evaluation-demo/src
cp .env.example ../rag-evaluation-demo/.env.example
```

### 4. Add secrets (Hugging Face Space Secrets)

In your Space settings → **Secrets**, add ONE of the following:

| Secret | Value | When |
|--------|-------|------|
| `OPENAI_API_KEY` | Your OpenAI key | If `LLM_PROVIDER=openai` |
| `GEMINI_API_KEY` | Your Gemini key | If `LLM_PROVIDER=gemini` |
| `LLM_PROVIDER` | `openai` or `gemini` or `mock` | Always |

> Without any key, the demo runs in **mock mode** (fully functional UI, deterministic responses).

### 5. Push to Spaces

```bash
cd ../rag-evaluation-demo
git add .
git commit -m "Initial deploy: Enterprise RAG Evaluation Engine Gradio demo"
git push
```

### 6. View your Space

URL: `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/rag-evaluation-demo`

---

## Notes

- The vector store (FAISS) is in-memory — documents ingested during a session are lost on restart.
- For persistent vector storage, configure `VECTOR_STORE_PATH` to a mounted volume.
- Free Spaces: 2 vCPU / 16 GB RAM — sufficient for FAISS similarity search with hundreds of chunks.
- For GPU-accelerated embedding (if you upgrade the embedding service), use a GPU Space.
