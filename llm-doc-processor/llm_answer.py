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
import time # Added for rate limiting
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from query_parser import ParsedQuery
from embedder import EmbeddedChunk
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _estimate_tokens(text: str) -> int:
    """A more standard heuristic to estimate token count (chars / 4)."""
    return int(len(text) / 4)

class DecisionEngine:
    """
    The DecisionEngine uses a generative AI model to make a final decision based
    on the user's query and the most relevant excerpts from policy documents.
    It constructs a detailed prompt to guide the LLM in its reasoning process
    and formats the output in a structured JSON format.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0, max_prompt_tokens: int = 128000):
        """
        Initializes the Decision Engine.

        Args:
            model_name: The name of the generative AI model to use.
            temperature: The creativity of the model's responses.
            max_prompt_tokens: The maximum number of tokens to allow in the prompt.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_prompt_tokens = max_prompt_tokens
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def make_decision_batch(self, processed_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes a single, batched decision for a list of questions with smart context stuffing.
        """
        logger.info(f"Making batched decision for {len(processed_questions)} questions.")
        prompt = self._construct_batch_prompt(processed_questions)
        
        if not prompt:
            return {"error": "Failed to construct a valid prompt from the provided context."}

        final_token_estimate = _estimate_tokens(prompt)
        logger.info(f"Final prompt token estimate: {final_token_estimate}")

        if final_token_estimate > self.max_prompt_tokens:
            logger.error(f"FATAL: Prompt size ({final_token_estimate}) exceeds max limit ({self.max_prompt_tokens}). Aborting API call.")
            return {"error": "Internal error: Prompt construction failed token validation."}

        try:
            response = self._call_gemini_with_retry(prompt)
            logger.info(f"LLM raw response: {response.text!r}")
            return self._parse_llm_batch_response(response.text)
        except Exception as e:
            logger.error(f"Error during LLM API call: {e}", exc_info=True)
            return {"error": "Failed to get a response from the language model.", "details": str(e)}

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 5, initial_delay: float = 1.0):
        """
        Calls the Gemini API with retry logic and exponential backoff.
        """
        for i in range(max_retries):
            try:
                logger.info(f"Attempt {i+1}/{max_retries} to call Gemini API.")
                response = self.model.generate_content(prompt)
                return response
            except Exception as e:
                logger.error(f"Gemini API call failed (Attempt {i+1}/{max_retries}): {e}")
                if "rate limit" in str(e).lower() or "resource exhausted" in str(e).lower() or "quota" in str(e).lower():
                    delay = initial_delay * (2 ** i) # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise # Re-raise other exceptions
        raise Exception(f"Failed to get a response from Gemini API after {max_retries} retries due to rate limit.")

    def _construct_batch_prompt(self, processed_questions: List[Dict[str, Any]]) -> str:
        """Constructs a single, detailed prompt for a batch of questions with token-aware context stuffing and Chain-of-Thought reasoning."""
        base_prompt_template = """You are an AI Insurance Policy Expert. Your task is to provide clear and accurate answers to a list of questions based *only* on the provided document excerpts.

**Your Reasoning Process (Chain-of-Thought):**

1.  **Analyze the Question:** For each question, understand its core intent.
2.  **Extract Key Sentences:** From the "Relevant Policy Clauses" provided for that question, identify and extract the exact sentences that directly answer the question.
3.  **Synthesize the Final Answer:** Based *only* on the key sentences you extracted, construct a final, concise answer. Keep the answer to a similar length as this example: 'A hospital is defined as an institution with at least 10 inpatient beds, qualified nursing staff, and a fully equipped operation theatre.'

**Final Output Format:**

Your final output *must* be a single, valid JSON object. This object must contain a single key, "answers", which is a list of strings. Each string in the list should be the synthesized answer to the corresponding question, in the order they were presented.

**Example JSON Output:**
```json
{{
  "answers": [
    "Answer to the first question based on extracted sentences.",
    "Answer to the second question based on extracted sentences."
  ]
}}
```

---

{questions_block}

---

Now, follow the reasoning process and provide the final answers in the specified JSON format.
"""
        
        questions_with_context = []
        # Account for the tokens in the template, minus the placeholder itself.
        current_tokens = _estimate_tokens(base_prompt_template.replace("{questions_block}", ""))
        separator = "\n\n---\n\n"
        separator_tokens = _estimate_tokens(separator)

        for i, item in enumerate(processed_questions):
            # Account for the separator tokens that will be added between question blocks.
            if i > 0:
                current_tokens += separator_tokens

            question_str = f"**Question {i+1}: {item['question']}**\n\nRelevant Policy Clauses:\n"
            question_tokens = _estimate_tokens(question_str)

            # Check if we can even fit the next question's header before proceeding.
            if current_tokens + question_tokens > self.max_prompt_tokens:
                logger.warning(f"Token limit reached. Cannot add question: {item['question']}")
                break

            current_tokens += question_tokens
            context_for_question = []
            max_chunks_per_question = 3 # Hard limit to reduce token count
            chunks_added = 0

            for chunk, score in item['context']:
                if chunks_added >= max_chunks_per_question:
                    logger.info(f"Reached max chunks ({max_chunks_per_question}) for this question. Skipping further chunks.")
                    break

                chunk_str = f"--- Chunk from {chunk.chunk.source_document} (Score: {score:.2f}) ---{chunk.chunk.content}"
                chunk_tokens = _estimate_tokens(chunk_str)
                if current_tokens + chunk_tokens <= self.max_prompt_tokens:
                    context_for_question.append(chunk_str)
                    current_tokens += chunk_tokens
                    chunks_added += 1
                else:
                    logger.info("Stopping context stuffing for this question to stay within token limit.")
                    break
            
            questions_with_context.append(question_str + "".join(context_for_question))

        if not questions_with_context:
            logger.error("Could not construct any questions within the token limit.")
            return None # Return None to indicate prompt construction failure

        all_questions_block = separator.join(questions_with_context)
        return base_prompt_template.format(questions_block=all_questions_block)

    def _parse_llm_batch_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the LLM's response to extract the structured JSON object with a list of answers."""
        try:
            # Find the JSON block within the markdown
            json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'({.*?})', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1).strip() # Clean the extracted JSON string
                decision = json.loads(json_str)
                if isinstance(decision.get('answers'), list):
                    logger.info(f"Successfully parsed {len(decision['answers'])} answers from batched response.")
                    return decision
                else:
                    raise json.JSONDecodeError("'answers' key is not a list.", json_str, 0)
            else:
                raise ValueError("Could not find a JSON block in the LLM response.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid JSON from LLM batch response: {e}")
            return {"error": "Could not parse LLM response.", "details": response_text}