from fastapi import FastAPI, HTTPException

from app.config import settings
from app.schemas import CandidateMatch, QueryRequest, QueryResponse
from app.vector_store import get_vector_store_stats


app = FastAPI(
    title="SmartTalent AI API",
    description="RAG-based CV analysis and candidate matching API.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "SmartTalent AI API is running.",
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
    }


@app.get("/health")
def health_check():
    try:
        vector_stats = get_vector_store_stats()

        return {
            "status": "healthy",
            "vector_store_ready": True,
            **vector_stats,
        }

    except Exception as error:
        return {
            "status": "degraded",
            "vector_store_ready": False,
            "detail": str(error),
        }


@app.post("/query", response_model=QueryResponse)
def query_candidates(request: QueryRequest):
    """
    Analyze a recruiter query and return the best matching candidate.

    The real RAG pipeline will be connected in later commits.
    """
    cleaned_query = request.query.strip()

    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return QueryResponse(
        answer="Candidate matching pipeline is not connected yet.",
        match=CandidateMatch(
            matched_index=-1,
            similarity_score=0.0,
            filename=None,
            preview=None,
        ),
    )
