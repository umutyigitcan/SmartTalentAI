# SmartTalentAI

SmartTalentAI is a RAG-powered CV analysis and candidate matching API built with FastAPI, FAISS, OpenAI embeddings, and GPT.

It helps recruiters search candidate CVs using natural language queries and returns AI-generated matching insights based on retrieved candidate data.

## Core Idea

The system converts CV documents into embeddings, stores them in a FAISS vector index, retrieves the most relevant candidate for a recruiter query, and uses GPT to explain why the candidate is a good match.

## Tech Stack

- Python
- FastAPI
- OpenAI API
- FAISS
- PyMuPDF
- NumPy
- Pydantic
- Uvicorn

## Privacy Note

Real CV files, FAISS indexes, and metadata files are excluded from the repository for privacy reasons.

Use your own sample PDF files locally inside:

data/pdfs/

## Status

SmartTalentAI is currently under development.
