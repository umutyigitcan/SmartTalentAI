from fastapi import FastAPI


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
    }
