
"""
API Routes Module
Defines the API endpoints for the document processing service.
"""

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import time
from typing import List, Optional
from embedder import DocumentEmbedder, EmbeddedChunk # Added for type hints

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Router ---
router = APIRouter()

# --- Request and Response Models ---
class HackRxRequest(BaseModel):
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]
from query_parser import QueryParser
from retriever import Retriever
from llm_answer import DecisionEngine

from reranker import ReRanker
from utils.index_utils import load_or_build_index # Import the new utility

# --- Global Components (initialized once at startup) ---
# These will be populated in the startup event
global_retriever: Optional[Retriever] = None
global_embedded_chunks: Optional[List[EmbeddedChunk]] = None
global_embedder: Optional[DocumentEmbedder] = None # Keep embedder for query embedding

decision_engine = DecisionEngine()
query_parser = QueryParser()
reranker = ReRanker()

# Placeholder for startup logic that will be moved to api.py
async def initialize_global_components():
    global global_retriever, global_embedded_chunks, global_embedder
    logger.info("Initializing global components: loading or building FAISS index and embeddings...")
    global_retriever, global_embedded_chunks = load_or_build_index()
    global_embedder = DocumentEmbedder() # Initialize embedder for query embedding
    logger.info("Global components initialized.")

# Note: The actual @app.on_event("startup") will be in api.py,
# and will call initialize_global_components.

# --- API Endpoint ---
import asyncio

import asyncio

@router.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def process_query(
    request_data: HackRxRequest,
    request: Request # Access to app.state
):
    """
    Process a document and a list of questions and return answers.
    """
    start_time = time.time()
    logger.info(f"Received request for {len(request_data.questions)} questions.")

    # Use globally initialized components
    retriever = global_retriever
    embedder = global_embedder
    embedded_chunks = global_embedded_chunks

    if not retriever or not embedder or not embedded_chunks:
        raise HTTPException(status_code=500, detail="Application not fully initialized. Please try again later.")

    top_k = int(os.getenv("TOP_K", 7))

    # 4. BATCH PROCESS ALL QUESTIONS
    # A. Parse all questions and embed them in a single batch
    all_questions = request_data.questions
    parsed_queries = [query_parser.parse_query(q) for q in all_questions]
    query_embeddings = embedder.embed_queries([pq.enhanced_query for pq in parsed_queries])

    # B. Concurrently retrieve a larger set of chunks for re-ranking
    initial_top_k = 20
    retrieval_tasks = [
        asyncio.to_thread(retriever.retrieve, pq, initial_top_k, emb)
        for pq, emb in zip(parsed_queries, query_embeddings)
    ]
    initial_retrievals = await asyncio.gather(*retrieval_tasks)

    # C. Concurrently re-rank the results for each question
    rerank_top_n = 7
    rerank_tasks = [
        asyncio.to_thread(reranker.rerank, pq, initial_ctx, rerank_top_n)
        for pq, initial_ctx in zip(parsed_queries, initial_retrievals)
    ]
    reranked_contexts = await asyncio.gather(*rerank_tasks)

    # D. Structure data for the decision engine
    processed_questions = [
        {"question": pq.original_query, "context": context}
        for pq, context in zip(parsed_queries, reranked_contexts)
    ]

    # 5. Make a single, batched decision with smart context stuffing
    decision_result = decision_engine.make_decision_batch(processed_questions)

    # 6. Format and return response
    if "error" in decision_result:
        raise HTTPException(status_code=500, detail=decision_result["error"])
    
    final_answers = decision_result.get("answers", [])
    if len(final_answers) != len(all_questions):
        logger.warning("Mismatch between number of questions and answers. Padding with default error.")
        final_answers.extend(["Could not generate an answer."] * (len(all_questions) - len(final_answers)))

    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return HackRxResponse(answers=final_answers)
