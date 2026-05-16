from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.analysis_service import analyze_candidate_match
from app.config import settings
from app.retrieval_service import get_best_candidate_match
from app.schemas import CandidateMatch, QueryRequest, QueryResponse
from app.vector_store import get_vector_store_stats, load_vector_store


app = FastAPI(
    title="SmartTalent AI API",
    description="RAG-based CV analysis and candidate matching API.",
    version="1.0.0",
)


origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://yigitumut.com",
    "https://www.yigitumut.com",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """
    Return API and vector store health information.
    """
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
    Analyze a recruiter query and return the best matching candidate with AI explanation.
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

        answer = analyze_candidate_match(
            query=cleaned_query,
            candidate=candidate,
        )

        return QueryResponse(
            answer=answer,
            match=CandidateMatch(
                matched_index=matched_index,
                similarity_score=similarity_score,
                filename=candidate.get("filename"),
                preview=candidate.get("preview"),
            ),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store is not ready: {str(error)}",
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store configuration error: {str(error)}",
        )

    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Candidate analysis failed: {str(error)}",
        )
