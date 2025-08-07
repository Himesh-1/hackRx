"""
API Routes Module - Minimal Fix Version
Defines the API endpoints for the document processing service.
"""

import os
import asyncio
import logging
import time
from typing import List, Any, Tuple, Union, Dict
import os
import asyncio
import logging
import time # Import time
from typing import List, Any, Tuple, Union, Dict
from fastapi import APIRouter, HTTPException, Request # Import Request
from pydantic import BaseModel

from loader import DocumentLoader
from chunker import DocumentChunker, Chunk
from embedder import DocumentEmbedder, EmbeddedChunk
from query_parser import QueryParser
from retriever import Retriever
from sparse_retriever import SparseRetriever
from reranker import ReRanker
from llm_answer import DecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Router ---
router = APIRouter()

# --- Request and Response Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@router.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def process_query(request_data: HackRxRequest, request: Request):
    """
    Process a document and a list of questions and return answers.
    """
    start_time = time.time()
    logger.info(f"Received request for document: {request_data.documents} and {len(request_data.questions)} questions.")

    try:
        # Access pre-loaded components from app.state
        dense_retriever = request.app.state.dense_retriever
        sparse_retriever = request.app.state.sparse_retriever
        embedded_chunks = request.app.state.embedded_chunks
        chunks = request.app.state.chunks # Original chunks for sparse retriever mapping

        if not dense_retriever or not sparse_retriever or not embedded_chunks or not chunks:
            raise HTTPException(status_code=500, detail="Application not fully initialized. Indices not loaded.")

        # 5. Process Questions
        process_questions_start = time.time()
        query_parser = QueryParser()
        reranker = ReRanker()
        decision_engine = DecisionEngine()

        top_k = int(os.getenv("TOP_K", 5))
        rerank_top_n = 5

        all_questions = request_data.questions
        query_parser_instance = QueryParser() # Instantiate QueryParser
        query_parser_instance = QueryParser() # Instantiate QueryParser
        parsed_queries = query_parser_instance.parse_queries(all_questions)
        
        # Embed queries using a fresh embedder instance if needed, or pass the one from app.state if it's a singleton
        # For now, let's assume embedder is stateless for query embedding and can be instantiated here
        query_embedder = DocumentEmbedder() 
        query_embeddings = query_embedder.embed_queries([pq.enhanced_query for pq in parsed_queries])

        # Concurrently retrieve from both dense and sparse retrievers
        retrieval_start = time.time()
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
        retrieval_end = time.time()
        logger.info(f"Step 5a: Retrieval (Dense + Sparse) took {retrieval_end - retrieval_start:.2f} seconds")

        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))

        # Filter dense results by similarity threshold
        filter_start = time.time()
        filtered_dense_results = []
        for res_list in dense_results:
            filtered_list = [(doc, score) for doc, score in res_list if score >= similarity_threshold]
            filtered_dense_results.append(filtered_list)
        filter_end = time.time()
        logger.info(f"Step 5b: Filtering Dense Results took {filter_end - filter_start:.2f} seconds")
        
        # Note: For sparse results (BM25), a direct similarity threshold like 0.6 might not be directly applicable
        # as BM25 scores are not normalized like cosine similarity. We'll keep them as is for now,
        # or a separate, context-specific threshold would be needed for BM25.

        # Fuse the results
        fuse_start = time.time()
        fused_results = []
        for dense_res, sparse_res in zip(filtered_dense_results, sparse_results):
            # Convert sparse results to the same format as dense results
            # Need to map sparse_res_content back to original Chunk objects
            sparse_res_chunks_with_objects = []
            for content, score in sparse_res:
                # Find the original chunk object for this content
                original_chunk = next((c for c in chunks if c.content == content), None)
                if original_chunk:
                    sparse_res_chunks_with_objects.append((original_chunk, score))
                else:
                    logger.warning(f"Original chunk object not found for content: {content[:50]}...")

            fused_results.append(_reciprocal_rank_fusion([dense_res, sparse_res_chunks_with_objects]))
        fuse_end = time.time()
        logger.info(f"Step 5c: Fusing Results took {fuse_end - fuse_start:.2f} seconds")

        # Re-rank the fused results
        rerank_start = time.time()
        rerank_tasks = [
            asyncio.to_thread(reranker.rerank, pq, fused_res_for_q, rerank_top_n)
            for pq, fused_res_for_q in zip(parsed_queries, fused_results)
        ]
        reranked_contexts = await asyncio.gather(*rerank_tasks)
        rerank_end = time.time()
        logger.info(f"Step 5d: Re-ranking took {rerank_end - rerank_start:.2f} seconds")

        # Generate answers
        llm_start = time.time()
        processed_questions = []
        for pq, context_list_with_scores in zip(parsed_queries, reranked_contexts):
            # context_list_with_scores now contains (Chunk/EmbeddedChunk, score) tuples
            # We need to pass the Chunk/EmbeddedChunk objects to the DecisionEngine
            processed_questions.append({
                "question": pq.original_query,
                "context": [item[0] for item in context_list_with_scores] # Pass the chunk object
            })

        try:
            decision_result = decision_engine.make_decisions_batch(processed_questions)
        except KeyError as e:
            logger.error(f"KeyError in decision engine - missing template variable: {e}")
            # Return default answers for all questions
            decision_result = {
                "answers": [{"answer": "Sorry, I cannot process this question due to a template error.", "source_clauses": []}] * len(all_questions)
            }
        except Exception as e:
            logger.error(f"Error in decision engine: {e}")
            decision_result = {
                "answers": [{"answer": "Sorry, I cannot process this question due to an internal error.", "source_clauses": []}] * len(all_questions)
            }
        llm_end = time.time()
        logger.info(f"Step 5e: LLM Answer Generation took {llm_end - llm_start:.2f} seconds")

        if "error" in decision_result:
            raise HTTPException(status_code=500, detail=decision_result["error"])

        final_answers = decision_result.get("answers", [])
        # Ensure all questions have an answer, even if empty
        if len(final_answers) != len(all_questions):
            logger.warning("Mismatch between number of questions and answers. Padding with default error.")
            while len(final_answers) < len(all_questions):
                final_answers.append("Could not generate an answer.")

        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def _reciprocal_rank_fusion(results_lists: List[List[Tuple[Any, float]]], k: int = 60) -> List[Tuple[Any, float]]:
    """
    Performs Reciprocal Rank Fusion on a list of ranked results.
    """
    fused_scores = {}
    for results in results_lists:
        for i, (doc, score) in enumerate(results):
            rank = i + 1
            
            # Use object id as fallback for chunk_id
            try:
                if hasattr(doc, 'chunk_id'):
                    current_chunk_id = doc.chunk_id
                elif hasattr(doc, 'chunk') and hasattr(doc.chunk, 'chunk_id'):
                    current_chunk_id = doc.chunk.chunk_id
                else:
                    current_chunk_id = id(doc)
            except Exception as e:
                logger.warning(f"Could not get chunk_id for document: {e}. Using object id.")
                current_chunk_id = id(doc)

            if current_chunk_id not in fused_scores:
                fused_scores[current_chunk_id] = {"score": 0, "doc": doc}
            fused_scores[current_chunk_id]["score"] += 1 / (k + rank)

    reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    return [(item["doc"], item["score"]) for item in reranked_results]