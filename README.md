# 3GPP RAG Chatbot

A Retrieval-Augmented Generation chatbot grounded in 3GPP TS 38.300 
(Release 19) with near-zero hallucination.

## Architecture

- **Knowledge base**: 3GPP TS 38.300 (5G NR overview spec)
- **Chunking**: 1000-char overlapping windows with 150-char overlap
- **Embedding model**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **Vector database**: ChromaDB with cosine similarity (HNSW index)
- **LLM**: Mistral-7B via Ollama (temperature=0.1)
- **Hallucination control**: confidence threshold + strict system prompt

## How to run

### 1. Install dependencies
pip install python-docx sentence-transformers chromadb flask

### 2. Install Ollama and pull Mistral
Download from ollama.ai, then:
ollama pull mistral

### 3. Build the knowledge base (one-time)
python src/ingest.py
python src/chunker.py
python src/embedder.py

### 4. Start the chatbot
python server.py
Open http://localhost:5000

## Anti-hallucination strategy

Four layers of protection:
1. Confidence threshold (0.40) — low similarity = "not found" response
2. LLM system prompt explicitly forbids using own training knowledge
3. Only retrieved 3GPP chunks are passed as context
4. Every answer includes source citations with similarity scores