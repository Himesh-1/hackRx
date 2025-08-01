"""
Decision Engine Module
Uses a Large Language Model (LLM) to make decisions based on retrieved context.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import numpy as np
from .query_parser import ParsedQuery
from .embedder import EmbeddedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Makes a decision using an LLM based on the query and retrieved document chunks.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
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

        Args:
            parsed_query: The ParsedQuery object.
            retrieved_chunks: A list of tuples containing retrieved chunks and their scores.

        Returns:
            A dictionary containing the decision, justification, and other relevant information.
        """
        logger.info(f"Making decision for query: '{parsed_query.original_query}'")

        prompt = self._construct_prompt(parsed_query, retrieved_chunks)

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            decision_json = response.text
            decision = json.loads(decision_json)
            logger.info(f"Decision made successfully: {decision.get('decision')}")
            return decision
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return {
                "error": "Failed to get a response from the language model.",
                "details": str(e)
            }

    def _construct_prompt(self, parsed_query: ParsedQuery, retrieved_chunks: List[Tuple[EmbeddedChunk, float]]) -> str:
        """
        Constructs the prompt for the LLM.
        """
        context = "\n".join([f"--- Chunk from {chunk.chunk.source_document} ---\n{chunk.chunk.content}" for chunk, score in retrieved_chunks])

        prompt = f"""
        Please analyze the following insurance query and the provided document excerpts to make a final decision.

        **Original Query:**
        {parsed_query.original_query}

        **Parsed Information:**
        - Intent: {parsed_query.intent}
        - Entities: {json.dumps(parsed_query.entities, indent=2)}

        **Relevant Document Excerpts:**
        {context}

        **Your Task:**
        Based on the query and the document excerpts, provide a JSON response with the following structure:
        {{
            "decision": "<Approve/Reject/Insufficient Information>",
            "justification": "<A clear and concise explanation for your decision, referencing specific document clauses if possible.>",
            "payout_amount": <The approved payout amount, or 0 if rejected>,
            "confidence_score": <A float between 0.0 and 1.0 indicating your confidence in the decision>,
            "referenced_clauses": [
                {{
                    "source_document": "<The name of the source document>",
                    "clause_text": "<The specific text of the referenced clause>"
                }}
            ]
        }}

        **Instructions:**
        - If the information is sufficient to approve the claim, set "decision" to "Approve".
        - If the information clearly indicates the claim should be rejected, set "decision" to "Reject".
        - If the provided excerpts are not sufficient to make a clear decision, set "decision" to "Insufficient Information".
        - The justification should be detailed and directly supported by the provided text.
        - The payout_amount should be based on the claim details and policy limits, if available.
        - Reference the specific clauses from the documents that support your decision.

        Provide only the JSON response.
        """
        return prompt

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This requires an OPENAI_API_KEY to be set in the environment.
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping DecisionEngine test: OPENAI_API_KEY not set.")
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
