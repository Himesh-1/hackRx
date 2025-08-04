
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

    # 1. Load document from URL
    try:
        document = loader.load_document_from_url(request_data.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document from URL: {e}")

    # 2. Chunk documents
    chunks = chunker.chunk_documents([document])

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from the document.")

    # 3. Embed chunks
    embedded_chunks = embedder.embed_chunks(chunks)

    if not embedded_chunks:
        raise HTTPException(status_code=400, detail="No embeddings generated from chunks.")

    # 4. Initialize Retriever with the current embedded chunks
    # Retriever needs to be initialized per request as it depends on the embedded_chunks
    retriever = Retriever(embedder, embedded_chunks)

    # Get TOP_K from environment variable or use default
    top_k = int(os.getenv("TOP_K", 3))

    answers = []
    for question in request_data.questions:
        logger.info(f"Processing question: {question}")
        # 1. Parse the query
        parsed_query = query_parser.parse_query(question)

        # 2. Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(parsed_query, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for question: {question}")
        for i, (chunk, score) in enumerate(retrieved_chunks):
            logger.debug(f"  Chunk {i+1} (Score: {score:.2f}): {chunk.chunk.content[:100]}...")

        if not retrieved_chunks:
            answers.append("Information not found in the document.")
            continue

        # 3. Make a decision
        decision_result = decision_engine.make_decision(parsed_query, retrieved_chunks)

        if "error" in decision_result:
            answers.append(decision_result["error"])
        else:
            answers.append(decision_result.get("answer", "Could not generate an answer."))

    return {"answers": answers}
