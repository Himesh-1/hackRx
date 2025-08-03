"""
Decision Engine Module
Uses a Large Language Model (LLM) to make decisions based on retrieved context.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import numpy as np
from .query_parser import ParsedQuery
from .embedder import EmbeddedChunk
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Makes a decision using an LLM based on the query and retrieved document chunks.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.1):
        """
        Initialize the Decision Engine.

        Args:
            model_name: The name of the OpenAI model to use.
            temperature: The creativity of the model's responses.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)

    def make_decision(self, parsed_query: ParsedQuery, retrieved_chunks: List[Tuple[EmbeddedChunk, float]]) -> Dict[str, Any]:
        """
        Makes a decision based on the parsed query and retrieved chunks.
        """
        logger.info(f"Making decision for query: '{parsed_query.original_query}'")
        prompt = self._construct_prompt(parsed_query, retrieved_chunks)

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            logger.info(f"Gemini raw response: {response.text!r}")

            if not response.text or not response.text.strip():
                return {"error": "LLM returned an empty response."}

            # Extract the JSON part from the response
            json_match = re.search(r'```json\n({.*?})\n```', response.text, re.DOTALL)
            if not json_match:
                # Fallback for responses that might not have the markdown
                json_match = re.search(r'({.*?})', response.text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                try:
                    decision = json.loads(json_str)
                    logger.info(f"Decision made successfully: {decision.get('decision')}")
                    return decision
                except json.JSONDecodeError as jde:
                    logger.error(f"JSON decode error in extracted string: {jde}")
                    return {"error": "LLM response did not contain valid JSON.", "details": json_str}
            else:
                logger.error("Could not find a JSON block in the LLM response.")
                return {"error": "Could not parse LLM response.", "details": response.text}

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            return {"error": "Failed to get a response from the language model.", "details": str(e)}

    def _construct_prompt(self, parsed_query: ParsedQuery, retrieved_chunks: List[Tuple[EmbeddedChunk, float]]) -> str:
        context = "\n".join([f"--- Chunk from {chunk.chunk.source_document} (Score: {score:.2f}) ---\n{chunk.chunk.content}" for chunk, score in retrieved_chunks])
        
        prompt = f"""**System Role:**
        You are an expert AI insurance claim adjudicator. Your task is to analyze insurance queries and make a definitive, evidence-based decision based *only* on the provided policy document excerpts.

        **Context:**
        - Today's Date: {datetime.now().strftime('%Y-%m-%d')}
        - Query Confidence: {parsed_query.confidence:.2f}

        **Original Query:**
        {parsed_query.original_query}

        **Parsed Information:**
        - Intent: {parsed_query.intent}
        - Entities: {json.dumps(parsed_query.entities, indent=2)}
        - Enhanced Query for Retrieval: {parsed_query.enhanced_query}

        **Relevant Policy Document Excerpts:**
        {context}

        **Your Task:**
        1.  **Chain of Thought:** First, reason through the query step-by-step. Explicitly state your reasoning process. Consider the query, the parsed entities, and each provided document chunk. Analyze if the chunks are relevant and sufficient. Identify the key clauses that support or deny the claim.
        2.  **Final Decision:** Based on your reasoning, provide a final JSON response with the specified structure. The decision must be directly supported by the text in the excerpts.

        **JSON Output Structure:**
        ```json
        {{
            "decision": "<Approved/Rejected/Insufficient Information>",
            "justification": "<A clear, concise, and evidence-based explanation for your decision, directly referencing specific document clauses and connecting them to the user's query.>",
            "payout_amount": <The approved payout amount as a float, or 0.0 if rejected or info is insufficient>,
            "confidence_score": <A float between 0.0 and 1.0 indicating your confidence in the decision based *only* on the provided information>,
            "referenced_clauses": [
                {{
                    "source_document": "<The name of the source document>",
                    "clause_text": "<The exact, verbatim text of the referenced clause>"
                }}
            ]
        }}
        ```

        **Instructions & Examples:**
        -   **Decision:** Must be one of `Approved`, `Rejected`, or `Insufficient Information`.
        -   **Justification:** Do not invent information. If a detail (e.g., age) is missing from the query but required by the policy, state it. Example: "The claim is rejected because the policy requires a minimum of 24 months of waiting for knee surgery, but the policy is only 3 months old as stated in Clause 12.1."
        -   **Payout Amount:** Only specify if the policy provides a clear amount for the procedure. Otherwise, default to 0.0.
        -   **Confidence Score:** High confidence (>0.9) for clear approve/reject cases. Medium (0.6-0.9) if there's some ambiguity. Low (<0.6) for insufficient information.
        -   **Insufficient Information:** Use this if a critical piece of information is missing from both the query and the documents to make a call. Explain what is missing in the justification.

        Begin your response with your chain of thought, followed by the final JSON object.
        """
        return prompt

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This requires a GEMINI_API_KEY to be set in the environment.
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping DecisionEngine test: GEMINI_API_KEY not set.")
    else:
        # 1. Initialize Decision Engine
        decision_engine = DecisionEngine()

        # 2. Create a mock parsed query and retrieved chunks
        mock_parsed_query = ParsedQuery(
            original_query="Is my knee surgery covered? I have a 3-month old policy.",
            enhanced_query="knee surgery coverage policy claim 3 month old",
            entities={'procedure': 'knee surgery', 'policy_duration': {'value': 3, 'unit': 'months'}},
            intent='claim_inquiry',
            keywords=['knee', 'surgery', 'coverage', 'policy'],
            confidence=0.95
        )

        # Mock retrieved chunks (in a real scenario, these come from the Retriever)
        from .chunker import Chunk
        mock_chunks = [
            (EmbeddedChunk(chunk=Chunk(content="Clause 12.1: A waiting period of 24 months is applicable for joint replacement surgeries.", chunk_id="doc1_chunk3", source_document="policy_document.pdf", chunk_index=3, start_char=1200, end_char=1300), embedding=np.random.rand(384), embedding_model="mock", embedding_dim=384), 0.95),
            (EmbeddedChunk(chunk=Chunk(content="General Exclusions: Pre-existing conditions are not covered in the first 48 months.", chunk_id="doc1_chunk8", source_document="policy_document.pdf", chunk_index=8, start_char=3400, end_char=3500), embedding=np.random.rand(384), embedding_model="mock", embedding_dim=384), 0.88)
        ]

        # 3. Make a decision
        decision_result = decision_engine.make_decision(mock_parsed_query, mock_chunks)

        print("\n--- Decision Engine Test Result ---")
        print(json.dumps(decision_result, indent=2))