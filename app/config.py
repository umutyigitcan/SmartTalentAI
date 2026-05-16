import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """
    Runtime configuration for SmartTalentAI.
    """

    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

    def validate_openai_settings(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env file.")


settings = Settings()
