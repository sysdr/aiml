import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = "get keys for aistudio.google.com"
    APP_NAME = "AI Chat Assistant"
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 8000
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

settings = Settings()
