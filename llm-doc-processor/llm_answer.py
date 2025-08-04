
"""
Decision Engine Module
This module uses a Large Language Model (LLM) to make decisions based on
retrieved context from policy documents. It is designed to provide detailed,
evidence-based answers to insurance-related queries.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from query_parser import ParsedQuery
from embedder import EmbeddedChunk
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    The DecisionEngine uses a generative AI model to make a final decision based
    on the user's query and the most relevant excerpts from policy documents.
    It constructs a detailed prompt to guide the LLM in its reasoning process
    and formats the output in a structured JSON format.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        """
        Initializes the Decision Engine.

        Args:
            model_name: The name of the generative AI model to use.
            temperature: The creativity of the model's responses. A lower value
                         is better for fact-based decisions.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def make_decision(self, parsed_query: ParsedQuery, retrieved_chunks: List[Tuple[EmbeddedChunk, float]]) -> Dict[str, Any]:
        """
        Makes a decision by sending a structured prompt to the LLM.

        Args:
            parsed_query (ParsedQuery): The parsed user query.
            retrieved_chunks (List[Tuple[EmbeddedChunk, float]]): A list of relevant document chunks and their scores.

        Returns:
            Dict[str, Any]: A dictionary containing the decision, justification, and other metadata.
        """
        logger.info(f"Making decision for query: '{parsed_query.original_query}'")
        prompt = self._construct_prompt(parsed_query, retrieved_chunks)
        logger.info(f"LLM input prompt: {prompt!r}")

        try:
            response = self.model.generate_content(prompt)
            logger.info(f"LLM raw response: {response.text!r}")

            return self._parse_llm_response(response.text)

        except Exception as e:
            logger.error(f"Error during LLM API call: {e}", exc_info=True)
            return {"error": "Failed to get a response from the language model.", "details": str(e)}

    def _construct_prompt(self, parsed_query: ParsedQuery, retrieved_chunks: List[Tuple[EmbeddedChunk, float]]) -> str:
        """Constructs a detailed, structured prompt for the LLM."""
        context = "\n".join([
            f"--- Chunk from {chunk.chunk.source_document} (Score: {score:.2f}) ---\n{chunk.chunk.content}"
            for chunk, score in retrieved_chunks
        ])
        
        return f"""You are an AI Insurance Policy Expert. Your primary role is to provide clear, comprehensive, and accurate information about insurance policies based on the provided documents.

---

User Query:
{parsed_query.original_query}

Rewritten Query for better understanding:
{parsed_query.enhanced_query}

---

Relevant Policy Clauses (retrieved using semantic similarity):
{context}

---

Your Task:

1.  **Analyze the User's Query:** Understand the user's intent, whether it's a specific claim scenario or a general question about policy coverage.

2.  **Synthesize Information:** Carefully review all the provided policy clauses. Synthesize the information to form a holistic understanding of the relevant terms, conditions, exclusions, and benefits.

3.  **Construct a Comprehensive Answer:** Based on your analysis, generate a detailed answer that directly addresses the user's query.

    *   **For specific claim scenarios:**
        *   Clearly state whether the claim is likely to be "Approved," "Rejected," or if "More Information is Needed."
        *   Provide a thorough justification for your decision, citing the specific clauses (including clause numbers or headings) that support your conclusion.
        *   If applicable, provide an estimated payout or benefit amount, explaining how you arrived at that figure based on the policy terms.

    *   **For general queries about the policy:**
        *   Provide a detailed explanation of the relevant policy sections.
        *   Clarify any complex terms or conditions.
        *   Use examples to illustrate how the policy works in practice.

4.  **Provide a Confidence Score:** Rate your confidence in the answer on a scale of 1 to 5, where 5 is very confident and 1 is not confident.

---

Rules:

*   **Clarity and Precision:** Your answer must be easy to understand, precise, and unambiguous.
*   **Evidence-Based:** Base your entire answer *only* on the provided policy clauses. Do not make assumptions or use external knowledge.
*   **Cite Your Sources:** Always reference the specific clause numbers or headings that support your statements.
*   **JSON Output:** Your final output must be a valid JSON object with the following structure:

```json
{{
  "summary": "A concise, one-sentence summary of the answer.",
  "answer": "A detailed and comprehensive answer to the user's query, following the guidelines above.",
  "confidence_score": <1-5>,
  "decision": "Approved" or "Rejected" or "More Information Needed" or "Not Applicable",
  "justification": "A detailed justification for the decision, citing relevant clauses. Null if not applicable.",
  "amount": <number>
}}
```
"""

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the LLM's response to extract the structured JSON decision."""
        # Find the JSON block within the markdown
        json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
        if not json_match:
            # Fallback for responses that might not have the markdown
            json_match = re.search(r'({.*?})', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                decision = json.loads(json_str)
                logger.info(f"Successfully parsed decision: {decision.get('decision')}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from LLM response: {e}")
                return {"error": "LLM response did not contain valid JSON.", "details": json_str}
        else:
            logger.error("Could not find a JSON block in the LLM response.")
            return {"error": "Could not parse LLM response.", "details": response_text}

# Example usage for testing purposes
if __name__ == '__main__':
    # This requires a GEMINI_API_KEY to be set in the environment.
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping DecisionEngine test: GEMINI_API_KEY not set.")
    else:
        decision_engine = DecisionEngine()
        mock_query = ParsedQuery(
            original_query="Is my knee surgery covered? My policy is 3 months old.",
            enhanced_query="knee surgery coverage policy claim 3 month old",
            entities={'procedure': 'knee surgery', 'policy_duration': {'value': 3, 'unit': 'months'}},
            intent='claim_inquiry',
            keywords=['knee', 'surgery', 'coverage', 'policy'],
            confidence=0.95
        )
        mock_chunks = [
            (EmbeddedChunk(chunk=None, embedding=None, embedding_model='', embedding_dim=0), 0.95),
            (EmbeddedChunk(chunk=None, embedding=None, embedding_model='', embedding_dim=0), 0.88)
        ]
        decision_result = decision_engine.make_decision(mock_query, mock_chunks)
        print("\n--- Decision Engine Test Result ---")
        print(json.dumps(decision_result, indent=2))
