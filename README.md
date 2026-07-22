# Research Paper Assistant

A production-oriented Retrieval-Augmented Generation (RAG) system that answers
questions over a corpus of research papers, with citation-grounded answers to
minimize hallucination.

Built to explore how RAG systems are actually engineered in production, not
just prototyped: modular architecture, automated tests, CI, containerized
deployment, and a retrieval-accuracy evaluation harness.

## Architecture

```
arXiv API ──> ingest.py ──> PDFs (data/papers/)
                                    │
                                    v
                          chunking.py (parse + sliding-window chunk)
                                    │
                                    v
                    vector_store.py (sentence-transformers embed + FAISS index)
                                    │
                                    v
                          data/index/ (persisted FAISS + metadata)
                                    │
                                    v
   user query ──> api.py ──> vector_store.search() ──> generate.py (Groq LLM)
                                    │
                                    v
                          cited, grounded answer
```

**Why this structure:** each stage (ingest, chunk, embed/retrieve, generate)
is a separate module with a single responsibility, so any piece can be
swapped independently — e.g. replacing FAISS with a hosted vector DB, or
Groq with a local model, requires touching only one file.

## Key design decisions

- **Sliding-window chunking with overlap** (`chunking.py`): fixed-size chunks
  with 150-word overlap, rather than per-page splitting, since paper sections
  frequently span page boundaries. Overlap avoids losing context at chunk
  edges — a common cause of poor retrieval.
- **Cosine similarity via normalized inner product** in FAISS (`IndexFlatIP`
  + L2 normalization), which is more standard for sentence-embedding
  similarity than raw L2 distance.
- **Grounded generation with explicit "I don't know"**: the system prompt in
  `generate.py` forces the model to admit when the retrieved context doesn't
  answer the question, instead of hallucinating.
- **Every answer cites source paper titles**, so answers are independently
  verifiable — an important property for anything research-adjacent.
- **Groq for inference**: free tier, fast (LPU-based inference), good enough
  quality for a portfolio project without requiring a paid API.

## Setup

```bash
git clone <your-repo-url>
cd research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your free Groq API key to .env (https://console.groq.com/keys)
```

## Usage

**1. Download papers from arXiv:**
```bash
python -m src.ingest --query "retrieval augmented generation" --max-results 30
```

**2. Build the vector index:**
```bash
python -m src.build_index
```

**3. Run the API:**
```bash
uvicorn src.api:app --reload
```
Visit `http://localhost:8000/docs` for interactive API docs.

**4. Query it:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG reduce hallucination compared to standard LLMs?"}'
```

## Running with Docker

```bash
docker compose up --build
```

## Testing

```bash
pytest tests/ -v
```
Tests cover chunking logic, the retrieval pipeline (build/save/load
round-trip), and API contract behavior — CI runs these on every push
(see `.github/workflows/ci.yml`).

## Evaluation

```bash
python -m eval.run_eval
```
Runs a labeled question set (`eval/eval_questions.json`) against the built
index and reports retrieval accuracy — add your own questions as you index
more papers, and track this number over time as you tune chunking/retrieval.

## Possible extensions

- Cross-encoder re-ranking of retrieved chunks before generation
- Hybrid search (BM25 + embeddings) for queries with exact technical terms
- Streaming responses from the LLM
- Swap FAISS for a hosted vector DB (Qdrant/Pinecone) for multi-user scale
