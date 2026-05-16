from fastapi import FastAPI, HTTPException

from app.config import settings
from app.retrieval_service import get_best_candidate_match
from app.schemas import CandidateMatch, QueryRequest, QueryResponse
from app.vector_store import get_vector_store_stats, load_vector_store


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
    """
    cleaned_query = request.query.strip()

    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        index, metadata = load_vector_store()
        matched_index, similarity_score, candidate = get_best_candidate_match(
            query=cleaned_query,
            index=index,
            metadata=metadata,
        )

        return QueryResponse(
            answer="Candidate retrieval completed. AI analysis will be added in the next step.",
            match=CandidateMatch(
                matched_index=matched_index,
                similarity_score=similarity_score,
                filename=candidate.get("filename"),
                preview=candidate.get("preview"),
            ),
        )

    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
