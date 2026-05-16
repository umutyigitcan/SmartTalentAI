from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Recruiter search query sent to the RAG candidate matching system.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language recruiter query such as required role, skills, or candidate profile.",
    )


class CandidateMatch(BaseModel):
    """
    Retrieved candidate metadata returned from the vector search step.
    """

    matched_index: int
    similarity_score: float
    filename: Optional[str] = None
    preview: Optional[str] = None


class QueryResponse(BaseModel):
    """
    Final response returned after retrieval and AI analysis.
    """

    answer: str
    match: CandidateMatch
