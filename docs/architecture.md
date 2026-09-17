# Architecture Notes

The public implementation follows the local RAG workflow used in the associated research:

1. Load a PDF knowledge source.
2. Split the document into overlapping chunks.
3. Convert chunks into dense semantic embeddings.
4. Store and search vectors with FAISS.
5. Maintain conversational state and structured project requirements.
6. Use the local Llama 3.1 8B model through Ollama.
7. Generate adaptive follow-up questions and a structured project summary.

The associated thesis also investigated a cloud deployment using Vertex AI and AlloyDB/pgvector. That infrastructure is intentionally outside this public repository because the original project included company-specific configuration and resources.
