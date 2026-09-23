# Retrieval-Augmented Generation (RAG) System Architecture

Retrieval-Augmented Generation (RAG) optimizes LLM outputs by retrieving authoritative facts from an external knowledge base before generating responses.

## Key System Components:
1. **Document Ingestion & Chunking**: Recursive character splitting with sliding window overlap maintains semantic continuity across text sections.
2. **Dense Vector Embeddings & Indexing**: Text representations are projected into a continuous vector space where cosine similarity indicates semantic closeness.
3. **Contextual Synthesis**: Injected context primes the foundational model with strict boundary rules to eliminate hallucinations.
4. **RAG Triad Automated Evaluation**:
   - **Faithfulness**: Measures if generated claims directly derive from context documents.
   - **Answer Relevance**: Measures whether response satisfies user intent.
   - **Context Precision**: Evaluates the signal-to-noise ratio of retrieved chunks.
