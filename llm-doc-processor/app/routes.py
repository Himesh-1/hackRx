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
    payout_amount: float
    confidence_score: float
    referenced_clauses: list

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
    logger.info(f"Received query: {query_request.query}")

    # 1. Parse the query
    query_parser = QueryParser()
    parsed_query = query_parser.parse_query(query_request.query)

    # 2. Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(parsed_query, top_k=5)

    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="Could not find any relevant information in the documents.")

    # 3. Make a decision
    decision = decision_engine.make_decision(parsed_query, retrieved_chunks)

    if "error" in decision:
        raise HTTPException(status_code=500, detail=decision["error"])

    return decision
