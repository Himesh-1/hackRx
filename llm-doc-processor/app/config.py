"""
Configuration Module
Contains settings and constants for the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Core Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DECISION_ENGINE_MODEL = "gemini-1.5-pro"

# --- Embedding Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_PROVIDER = "sentence_transformers"

# --- Retriever Configuration ---
TOP_K_RETRIEVAL = 5

# --- API Configuration ---
API_TITLE = "LLM Document Processor API"
API_VERSION = "1.0.0"
