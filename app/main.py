from fastapi import FastAPI

from app.config import settings


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
