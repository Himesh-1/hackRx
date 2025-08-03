"""
API Routes Module
Defines the API endpoints for the document processing service.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from models.query_parser import QueryParser
from models.retriever import Retriever
from models.decision_engine import DecisionEngine
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

# --- API Router ---
router = APIRouter()

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    query: str

class DecisionResponse(BaseModel):
    decision: str
    justification: str
    amount: float
    confidence: float
    clauses: list
    processing_time: float

# --- Dependency Injection --- 
def get_retriever(request: Request) -> Retriever:
    if not request.app.state.retriever:
        raise HTTPException(status_code=503, detail="Retriever is not available. Check if documents were loaded correctly.")
    return request.app.state.retriever

def get_decision_engine(request: Request) -> DecisionEngine:
    if not request.app.state.decision_engine:
        raise HTTPException(status_code=503, detail="Decision engine is not available.")
    return request.app.state.decision_engine

# --- API Endpoint ---
@router.post("/process-query", response_model=DecisionResponse)
def process_query(
    query_request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    decision_engine: DecisionEngine = Depends(get_decision_engine)
):
    """
    Process a natural language query and return a decision.
    """
    start_time = time.time()
    logger.info(f"Received query: {query_request.query}")

    # 1. Parse the query
    query_parser = QueryParser()
    parsed_query = query_parser.parse_query(query_request.query)

    # 2. Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(parsed_query, top_k=5)

    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="Could not find any relevant information in the documents.")

    # 3. Make a decision
    decision_result = decision_engine.make_decision(parsed_query, retrieved_chunks)

    if "error" in decision_result:
        raise HTTPException(status_code=500, detail=decision_result["error"])

    end_time = time.time()
    processing_time = end_time - start_time

    # Map backend fields to frontend fields
    response_data = {
        "decision": decision_result.get("decision"),
        "justification": decision_result.get("justification"),
        "amount": decision_result.get("payout_amount"),
        "confidence": decision_result.get("confidence_score"),
        "clauses": decision_result.get("referenced_clauses"),
        "processing_time": processing_time
    }

    return response_data
