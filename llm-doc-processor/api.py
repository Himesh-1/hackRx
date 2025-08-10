
"""
Main Application Module
Initializes the FastAPI application and the core processing components.
"""
import json
from pyngrok import ngrok
from fastapi import FastAPI
from routes import router
from llm_answer import DecisionEngine
import logging
from utils.index_manager import load_or_build_persistent_indices, DOC_HASHES_FILE # Import the index manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router ---
app.include_router(router)

from database_utils import initialize_db

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing database...")
    initialize_db()
    logger.info("Application startup: Loading or building persistent indices...")
    app.state.dense_retriever, \
    app.state.sparse_retriever, \
    app.state.embedded_chunks, \
    app.state.chunks = load_or_build_persistent_indices()
    with open(DOC_HASHES_FILE, "r") as f:
        app.state.doc_hashes = json.load(f)
    logger.info("Application startup: Persistent indices and document hashes loaded/built.")



