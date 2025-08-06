
"""
API Routes Module
Defines the API endpoints for the document processing service.
"""

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import time
from typing import List, Optional, Any
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

from sparse_retriever import SparseRetriever
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

global_sparse_retriever: Optional[SparseRetriever] = None

# Placeholder for startup logic that will be moved to api.py
async def initialize_global_components():
    global global_retriever, global_embedded_chunks, global_embedder, global_sparse_retriever
    logger.info("Initializing global components: loading or building FAISS index and embeddings...")
    global_retriever, global_embedded_chunks = load_or_build_index()
    global_embedder = DocumentEmbedder() # Initialize embedder for query embedding
    global_sparse_retriever = SparseRetriever([chunk.chunk for chunk in global_embedded_chunks])
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
    dense_retriever = global_retriever
    sparse_retriever = global_sparse_retriever
    embedder = global_embedder

    if not dense_retriever or not sparse_retriever or not embedder:
        raise HTTPException(status_code=500, detail="Application not fully initialized. Please try again later.")

    top_k = int(os.getenv("TOP_K", 10))

    # A. Parse all questions and embed them in a single batch
    all_questions = request_data.questions
    parsed_queries = [query_parser.parse_query(q) for q in all_questions]
    query_embeddings = embedder.embed_queries([pq.enhanced_query for pq in parsed_queries])

    

    # B. Concurrently retrieve from both dense and sparse retrievers
    dense_retrieval_tasks = [
        asyncio.to_thread(dense_retriever.retrieve, pq, top_k, emb)
        for pq, emb in zip(parsed_queries, query_embeddings)
    ]
    sparse_retrieval_tasks = [
        asyncio.to_thread(sparse_retriever.retrieve, pq, top_k)
        for pq in parsed_queries
    ]

    dense_results = await asyncio.gather(*dense_retrieval_tasks)
    sparse_results = await asyncio.gather(*sparse_retrieval_tasks)

    # C. Fuse the results using Reciprocal Rank Fusion (RRF)
    fused_results = []
    for dense_res, sparse_res in zip(dense_results, sparse_results):
        fused_results.append(_reciprocal_rank_fusion([dense_res, sparse_res]))

    # D. Concurrently re-rank the fused results for each question
    rerank_top_n = 5
    rerank_tasks = [
        asyncio.to_thread(reranker.rerank, pq, fused_res, rerank_top_n)
        for pq, fused_res in zip(parsed_queries, fused_results)
    ]
    reranked_contexts = await asyncio.gather(*rerank_tasks)

    # E. Make decisions for each question in a batch
    processed_questions = [
        {"question": pq.original_query, "context": context}
        for pq, context in zip(parsed_queries, reranked_contexts)
    ]
    decision_result = decision_engine.make_decisions_batch(processed_questions)

    # F. Format and return response
    if "error" in decision_result:
        raise HTTPException(status_code=500, detail=decision_result["error"])
    
    final_answers = decision_result.get("answers", [])
    if len(final_answers) != len(all_questions):
        logger.warning("Mismatch between number of questions and answers. Padding with default error.")
        final_answers.extend(["Could not generate an answer."] * (len(all_questions) - len(final_answers)))

    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return HackRxResponse(answers=final_answers)

def _reciprocal_rank_fusion(results_lists: List[List[tuple[Any, float]]], k: int = 60) -> List[tuple[Any, float]]:
    """
    Performs Reciprocal Rank Fusion on a list of ranked results.
    """
    fused_scores = {}
    for results in results_lists:
        for i, (doc, score) in enumerate(results):
            rank = i + 1
            # Safely get chunk_id from either EmbeddedChunk or Chunk
            if hasattr(doc, 'chunk_id'): # For Chunk objects
                current_chunk_id = doc.chunk_id
            elif hasattr(doc, 'chunk') and hasattr(doc.chunk, 'chunk_id'): # For EmbeddedChunk objects
                current_chunk_id = doc.chunk.chunk_id
            else:
                logger.warning(f"Document object {doc} has no identifiable chunk_id. Skipping.")
                continue

            if current_chunk_id not in fused_scores:
                fused_scores[current_chunk_id] = {"score": 0, "doc": doc}
            fused_scores[current_chunk_id]["score"] += 1 / (k + rank)

    reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    return [(item["doc"], item["score"]) for item in reranked_results]

    

    
