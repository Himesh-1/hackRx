"""
API Routes Module - Minimal Fix Version
Defines the API endpoints for the document processing service.
"""

import os
import asyncio
import logging
import time
from typing import List, Any, Tuple, Union, Dict
import hashlib
import json
import numpy as np
from database_utils import log_question
from fastapi import APIRouter, HTTPException, Request # Import Request
from pydantic import BaseModel

from utils import cache_utils # Added for caching LLM responses

from loader import DocumentLoader
from chunker import DocumentChunker, Chunk
from embedder import DocumentEmbedder, EmbeddedChunk
from query_parser import QueryParser
from retriever import Retriever
from sparse_retriever import SparseRetriever
from reranker import ReRanker
from llm_answer import DecisionEngine, _estimate_tokens

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
        doc_hashes = request.app.state.doc_hashes

        if not dense_retriever or not sparse_retriever or not embedded_chunks or not chunks:
            raise HTTPException(status_code=500, detail="Application not fully initialized. Indices not loaded.")

        # Download the document and compute its hash
        loader = DocumentLoader()
        try:
            downloaded_doc = await loader.load_document_from_url(request_data.documents)
            with open(loader.download_dir / downloaded_doc.filename, "rb") as f:
                doc_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download or hash document: {e}")

        # Check if the document is already processed
        if downloaded_doc.filename in doc_hashes and doc_hashes[downloaded_doc.filename] == doc_hash:
            logger.info(f"Document {downloaded_doc.filename} is already processed. Using pre-built indices.")
            # Filter chunks for the matched document
            doc_chunks = [chunk for chunk in chunks if chunk.source_document == downloaded_doc.filename]
            doc_embedded_chunks = [emb_chunk for emb_chunk in embedded_chunks if emb_chunk.chunk.source_document == downloaded_doc.filename]
        else:
            logger.info(f"Document {downloaded_doc.filename} is new. Processing from scratch.")
            # Process the new document
            chunker = DocumentChunker()
            doc_chunks = chunker.chunk_document(downloaded_doc)
            embedder = DocumentEmbedder()
            doc_embedded_chunks = embedder.embed_chunks(doc_chunks)

            # Update the global state with the new data
            embedded_chunks.extend(doc_embedded_chunks)
            chunks.extend(doc_chunks)
            doc_hashes[downloaded_doc.filename] = doc_hash

            # Update the retrievers with the new data
            dense_retriever.index.add(np.array([chunk.embedding for chunk in doc_embedded_chunks]).astype('float32'))
            # The sparse retriever needs to be re-initialized with the new corpus
            new_corpus = [c.content for c in chunks]
            request.app.state.sparse_retriever = SparseRetriever(new_corpus)

            # Persist the updated data
            from utils.index_manager import build_persistent_indices
            build_persistent_indices()

        # 5. Process Questions
        process_questions_start = time.time()
        query_parser = QueryParser()
        reranker = ReRanker()
        decision_engine = DecisionEngine()

        top_k = int(os.getenv("TOP_K", 20))
        rerank_top_n = 10

        all_questions = request_data.questions
        query_parser_instance = QueryParser() # Instantiate QueryParser
        
        # --- BATCHED PIPELINE WITH PROPER CACHE AND ANSWER EXTRACTION ---
        processed_questions = []
        cache_keys = []
        temp_engine = DecisionEngine()
        for question_text in all_questions:
            # Build context for each question
            parsed_query = query_parser_instance.parse_queries([question_text])[0]
            query_embedder = DocumentEmbedder()
            query_embedding = query_embedder.embed_queries([parsed_query.enhanced_query])[0]
            dense_result = await asyncio.to_thread(dense_retriever.retrieve, parsed_query, top_k, query_embedding)
            sparse_result = await asyncio.to_thread(request.app.state.sparse_retriever.retrieve, parsed_query, top_k)
            sparse_res_chunks_with_objects = []
            for content, score in sparse_result:
                original_chunk = next((c for c in doc_chunks if c.content == content), None)
                if original_chunk:
                    sparse_res_chunks_with_objects.append((original_chunk, score))
            fused_result = _reciprocal_rank_fusion([dense_result, sparse_res_chunks_with_objects])
            reranked_context = reranker.rerank([parsed_query], [fused_result], rerank_top_n)[0]
            processed_questions.append({
                "question": parsed_query.original_query,
                "context": [item[0] for item in reranked_context]
            })
            # Compute cache key as in llm_answer.py
            prompt = temp_engine._construct_batch_prompt([{
                "question": parsed_query.original_query,
                "context": [item[0] for item in reranked_context]
            }])
            cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest() if prompt else None
            cache_keys.append(cache_key)

        # Check cache for all questions
        answers = [None] * len(all_questions)
        for i, cache_key in enumerate(cache_keys):
            if cache_key:
                cached = cache_utils.get_cache(cache_key)
                if cached and isinstance(cached, dict) and isinstance(cached.get("answers"), list) and len(cached["answers"]) == 1:
                    ans = cached["answers"][0]
                    if not (isinstance(ans, str) and (ans.startswith("Error:") or "Could not generate an answer" in ans or "Sorry, I cannot process" in ans)):
                        answers[i] = ans

        # Only send missing/failed questions to LLM
        questions_to_send = [processed_questions[i] for i, a in enumerate(answers) if a is None]
        idxs_to_fill = [i for i, a in enumerate(answers) if a is None]
        if questions_to_send:
            decision_result = decision_engine.make_decisions_batch(questions_to_send)
            new_answers = decision_result.get("answers", [])
            for j, idx in enumerate(idxs_to_fill):
                ans = new_answers[j] if j < len(new_answers) else "Error: No answer generated."
                answers[idx] = ans
                # Set cache for this answer using the same cache key
                cache_key = cache_keys[idx]
                if cache_key and not (isinstance(ans, str) and (ans.startswith("Error:") or "Could not generate an answer" in ans or "Sorry, I cannot process" in ans)):
                    cache_utils.set_cache(cache_key, {"answers": [ans]})
        # Log and return
        for i, question_text in enumerate(all_questions):
            answer = answers[i]
            answered_correctly = not ("Sorry, I cannot process this question" in answer or "Could not generate an answer." in answer)
            log_question(question_text, request_data.documents, answered_correctly)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return HackRxResponse(answers=answers)

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
