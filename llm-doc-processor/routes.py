
"""
API Routes Module
Defines the API endpoints for the document processing service.
"""

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from query_parser import QueryParser
from retriever import Retriever
from llm_answer import DecisionEngine
from loader import DocumentLoader
from chunker import DocumentChunker
from embedder import DocumentEmbedder
import logging
import time
from typing import List, Optional

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
class HackRxResponse(BaseModel):
    answers: List[str]

"""
API Routes Module
Defines the API endpoints for the document processing service.
"""

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from query_parser import QueryParser
from retriever import Retriever
from llm_answer import DecisionEngine
from loader import DocumentLoader
from chunker import DocumentChunker
from embedder import DocumentEmbedder
import logging
import time
from typing import List, Optional

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

# --- Initialize global components ---
# These components are initialized once when the application starts
# and reused across all requests to improve performance and reduce overhead.
loader = DocumentLoader()
chunker = DocumentChunker()
embedder = DocumentEmbedder()
decision_engine = DecisionEngine()
query_parser = QueryParser()

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
    logger.info(f"Received request for document: {request_data.documents}")

    if not request_data.questions:
        return HackRxResponse(answers=[])

    # 1. Load document from URL
    try:
        document = loader.load_document_from_url(request_data.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document from URL: {e}")

    # 2. Chunk and Embed Document
    chunks = chunker.chunk_documents([document])
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from the document.")
    embedded_chunks = embedder.embed_chunks(chunks)
    if not embedded_chunks:
        raise HTTPException(status_code=400, detail="No embeddings generated from chunks.")

    # 3. Initialize Retriever
    retriever = Retriever(embedder, embedded_chunks)
    top_k = int(os.getenv("TOP_K", 7))

    # 4. BATCH PROCESS ALL QUESTIONS
    # A. Parse all questions and embed them in a single batch
    all_questions = request_data.questions
    parsed_queries = [query_parser.parse_query(q) for q in all_questions]
    query_embeddings = embedder.embed_queries([pq.enhanced_query for pq in parsed_queries])

    # B. Concurrently retrieve chunks for all questions in separate threads
    retrieval_tasks = [
        asyncio.to_thread(retriever.retrieve, pq, top_k, emb)
        for pq, emb in zip(parsed_queries, query_embeddings)
    ]
    retrieved_contexts = await asyncio.gather(*retrieval_tasks)

    # C. Structure data for the decision engine
    processed_questions = [
        {"question": pq.original_query, "context": context}
        for pq, context in zip(parsed_queries, retrieved_contexts)
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
