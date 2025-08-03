"""
Main Application Module
Initializes the FastAPI application and the core processing components.
"""

from fastapi import FastAPI
from . import config
from .routes import router
from models.document_loader import DocumentLoader
from models.chunker import DocumentChunker
from models.embedder import DocumentEmbedder
from models.retriever import Retriever
from models.decision_engine import DecisionEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Initialization ---
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Components ---
# These components will be initialized once at startup.
app.state.retriever = None
app.state.decision_engine = None

@app.on_event("startup")
def startup_event():
    """Actions to perform on application startup."""
    logger.info("Application startup: Initializing components...")

    # 1. Load documents
    loader = DocumentLoader(docs_path=str(config.DOCS_DIR))
    documents = loader.load_all_documents()

    if not documents:
        logger.warning("No documents found. The API will run but may not be able to process queries.")
        return

    # 2. Chunk documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    # 3. Embed chunks
    embedder = DocumentEmbedder(
        model_name=config.EMBEDDING_MODEL,
        embedding_provider=config.EMBEDDING_PROVIDER,
        cache_dir=str(config.EMBEDDINGS_DIR)
    )
    embedded_chunks = embedder.embed_chunks(chunks)

    # 4. Initialize Retriever
    app.state.retriever = Retriever(embedder, embedded_chunks)

    # 5. Initialize Decision Engine
    app.state.decision_engine = DecisionEngine(
        model_name=config.DECISION_ENGINE_MODEL
    )

    logger.info("All components initialized successfully.")

# --- API Router ---
app.include_router(router)
