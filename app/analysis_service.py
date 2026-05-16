from typing import Any, Dict

from app.config import settings
from app.embedding_service import get_openai_client


SYSTEM_PROMPT = (
    "You are an AI assistant that analyzes CVs for a recruiter. "
    "Given a role or skill query and candidate information, select the most relevant "
    "candidate and clearly explain why they are a good match. "
    "Focus on skills, experience, technologies, projects, and domain fit. "
    "Be specific, concise, and professional."
)


def build_candidate_analysis_prompt(query: str, candidate: Dict[str, Any]) -> str:
    """
    Build the user prompt for candidate match analysis.
    """
    filename = candidate.get("filename", "unknown")
    preview = candidate.get("preview", "")

    return (
        f"Recruiter query:\n{query}\n\n"
        f"Candidate file:\n{filename}\n\n"
        f"Candidate CV content:\n{preview}\n\n"
        "Based on this information, explain why this candidate is a good match. "
        "Mention relevant skills, experience, technologies, and possible fit."
    )


def analyze_candidate_match(query: str, candidate: Dict[str, Any]) -> str:
    """
    Generate a recruiter-focused candidate match explanation using OpenAI.
    """
    client = get_openai_client()
    prompt = build_candidate_analysis_prompt(query, candidate)

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content or ""
