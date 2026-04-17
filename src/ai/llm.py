import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def call_google_generative_ai_model():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    return model