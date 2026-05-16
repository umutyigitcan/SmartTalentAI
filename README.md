# SmartTalentAI

SmartTalentAI is a RAG-based CV analysis and candidate matching backend.

It extracts text from CV PDF files, generates OpenAI embeddings, stores vectors in a FAISS index, keeps metadata records, retrieves the most relevant candidate for a recruiter query, and generates a professional AI explanation about why the candidate fits the role.

This is not a simple AI wrapper. It is a modular backend system combining PDF parsing, embedding generation, FAISS vector search, metadata persistence, RAG retrieval, and OpenAI-powered recruiter analysis.

## Features

- Extracts text from CV PDF files
- Generates embeddings with OpenAI
- Normalizes vectors for similarity search
- Builds and saves a FAISS vector index
- Stores CV metadata as JSON
- Loads FAISS index and metadata at runtime
- Retrieves the most relevant candidate for a recruiter query
- Uses OpenAI to generate recruiter-focused candidate analysis
- Provides FastAPI endpoints for health checks and candidate querying
- Includes CORS configuration
- Handles empty queries, missing vector store files, no matches, and AI errors

## Tech Stack

- Python
- FastAPI
- OpenAI API
- FAISS
- PyMuPDF
- NumPy
- Pydantic
- Uvicorn
- RAG
- Vector Search

## Project Structure

```text
SmartTalentAI/
  app/
    __init__.py
    main.py
    config.py
    schemas.py
    pdf_loader.py
    embedding_service.py
    index_builder.py
    metadata_store.py
    vector_store.py
    retrieval_service.py
    analysis_service.py
  data/
    .gitkeep
  .env.example
  .gitignore
  requirements.txt
  README.md
```

## Environment Variables

Create a `.env` file based on `.env.example`.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare CV Data

Real CV files are excluded from this repository for privacy reasons.

Add your local sample CV PDF files under:

```text
data/pdfs/
```

Then build the FAISS index and metadata:

```bash
python -m app.index_builder
```

This creates the following generated files:

```text
data/rag_index.faiss
data/metadata_list.json
```

These generated files should not be committed to GitHub.

## Run the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Default local API URL:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### GET /

Basic API status endpoint.

Example response:

```json
{
  "status": "ok",
  "message": "SmartTalent AI API is running.",
  "chat_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

### GET /health

Returns vector store health and metadata statistics.

Example response:

```json
{
  "status": "healthy",
  "index_vectors": 4,
  "metadata_count": 4
}
```

### POST /query

Finds the best matching candidate for a recruiter query and generates an AI-powered explanation.

Request example:

```json
{
  "query": "Find the best candidate for a Python backend developer role with FastAPI and AI experience."
}
```

Response example:

```json
{
  "answer": "The selected candidate is a strong match because they have relevant Python backend experience, FastAPI knowledge, and AI project background.",
  "match": {
    "matched_index": 0,
    "similarity_score": 0.82,
    "filename": "candidate.pdf",
    "preview": "Candidate CV text preview..."
  }
}
```

## Example cURL Request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Find a candidate with Python, FastAPI, AI and backend development experience."}'
```

## How It Works

SmartTalentAI follows a Retrieval-Augmented Generation pipeline:

1. CV PDF files are placed inside `data/pdfs/`.
2. The PDF loader extracts text from each CV.
3. OpenAI embeddings are generated from CV text.
4. Embeddings are normalized for similarity search.
5. FAISS stores the vector index locally.
6. Metadata is saved as JSON.
7. A recruiter sends a query to the `/query` endpoint.
8. The query is embedded and compared against the FAISS index.
9. The most relevant candidate is retrieved.
10. OpenAI generates a professional recruiter-focused explanation.

## Privacy Note

Real CV files, generated FAISS indexes, and metadata files are excluded from the repository for privacy reasons.

Do not commit:

```text
.env
data/pdfs/
data/rag_index.faiss
data/metadata_list.json
```

Only source code, safe configuration templates, and documentation should be committed.

## Error Handling

SmartTalentAI includes error handling for:

- Empty user queries
- Missing OpenAI API key
- Missing FAISS index file
- Missing metadata file
- No matching candidate found
- Vector store loading errors
- OpenAI analysis failures

## Portfolio Value

This project demonstrates practical backend AI engineering skills:

- RAG architecture
- Vector search
- CV parsing
- Embedding generation
- FAISS indexing
- Metadata persistence
- FastAPI backend development
- OpenAI API integration
- Recruiter-focused AI analysis
- Production-style modular code organization

## Status

SmartTalentAI is portfolio-ready as a backend RAG system.

Possible future improvements:

- Multi-candidate ranking
- Authentication
- Admin dashboard
- Database persistence
- Uploaded CV management
- Candidate comparison reports
- PDF preview from frontend
- Frontend dashboard integration