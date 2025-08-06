
"""
Main Application Module
Initializes the FastAPI application and the core processing components.
"""
from pyngrok import ngrok
from fastapi import FastAPI
from routes import router, initialize_global_components # Import initialize_global_components
from llm_answer import DecisionEngine
import logging

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

# --- Global Components (if any, for shared resources not tied to request) ---
# For now, DecisionEngine is initialized per request in routes.py
# If there are truly global, long-lived resources, they can be initialized here.
# For example, if the LLM model itself needs to be loaded once.

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing global components...")
    await initialize_global_components() # Call the initialization function
    logger.info("Application startup: Global components initialized.")

