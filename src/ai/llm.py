import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def call_google_generative_ai_model(model_name: str = "gemini-2.5-flash-lite") -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance bound to the specified model."""
    logger.debug(f"Initialising {model_name} model")
    model = ChatGoogleGenerativeAI(model=model_name)
    return model