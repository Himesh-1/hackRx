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
from chunker import Chunk # Import Chunk to handle both types
from datetime import datetime
from dotenv import load_dotenv
from utils.cache_utils import get_cache, set_cache, delete_cache # Import cache utilities
import hashlib # Import hashlib for cache key generation

# Load environment variables
load_dotenv()

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

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0, max_prompt_tokens: int = 256000):
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
        self.api_keys = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(',') if key.strip()]
        if not self.api_keys:
            raise ValueError("GEMINI_API_KEYS environment variable not set or is empty.")
        self.current_key_index = 0
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(self.model_name)
        self.prompt_template_path = os.getenv("LLM_PROMPT_TEMPLATE", "prompts/decision_prompt.txt")
        self._load_prompt_template()

    def _load_prompt_template(self):
        """Loads the prompt template from a file."""
        try:
            with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                self.base_prompt_template = f.read()
            logger.info(f"Loaded prompt template from {self.prompt_template_path}")
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {self.prompt_template_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = int(os.getenv("LLM_MAX_RETRIES", 7)), initial_delay: float = 1.0):
        """
        Calls the Gemini API with retry logic and exponential backoff.
        """
        for i in range(max_retries):
            try:
                # Configure with the current API key before each attempt
                genai.configure(api_key=self.api_keys[self.current_key_index])
                logger.info(f"Attempt {i+1}/{max_retries} with key index {self.current_key_index} to call Gemini API.")
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 2048)) # Increased default for full JSON
                    )
                )
                return response
            except Exception as e:
                logger.error(f"Gemini API call failed (Attempt {i+1}/{max_retries}): {e}")
                if "rate limit" in str(e).lower() or "resource exhausted" in str(e).lower() or "quota" in str(e).lower():
                    logger.warning("Rate limit hit. Attempting to switch API key.")
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    logger.warning(f"Switched to key index: {self.current_key_index}")
                    delay = initial_delay * (2 ** i) # Exponential backoff
                    logger.warning(f"Retrying in {delay:.2f} seconds... (Attempt {i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise # Re-raise other exceptions
        raise Exception(f"Failed to get a response from Gemini API after {max_retries} retries across all keys due to rate limit or other persistent error.")

    def make_decisions_batch(self, processed_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes a single, batched decision for a list of questions with smart context stuffing.
        Returns a dictionary with a list of answer strings.
        """
        logger.info(f"Making batched decision for {len(processed_questions)} questions.")
        
        all_answers = []
        llm_batch_size = int(os.getenv("LLM_BATCH_SIZE", 5)) # Default to 5 questions per LLM call
        key_switch_interval = int(os.getenv("KEY_SWITCH_INTERVAL", 10)) # Switch key after 10 questions
        questions_processed_in_session = 0

        for i in range(0, len(processed_questions), llm_batch_size):
            batch_questions = processed_questions[i:i + llm_batch_size]
            logger.info(f"Processing batch of {len(batch_questions)} questions (questions {i+1} to {min(i+llm_batch_size, len(processed_questions))}).")

            # Proactive key rotation
            if questions_processed_in_session >= key_switch_interval:
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                logger.info(f"Proactively switched to API key index: {self.current_key_index}")
                questions_processed_in_session = 0 # Reset counter for new key

            prompt = self._construct_batch_prompt(batch_questions)
            
            if not prompt:
                # If prompt construction fails for a batch, append errors for these questions
                all_answers.extend(["Error: Failed to construct prompt for this batch."] * len(batch_questions))
                continue

            # Generate a cache key from the prompt
            cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            cached_response = get_cache(cache_key)
            if cached_response:
                # Validate cached response structure
                if isinstance(cached_response, dict) and isinstance(cached_response.get("answers"), list):
                    # Further validate that all items in 'answers' are strings
                    if all(isinstance(ans, str) for ans in cached_response["answers"]):
                        logger.info(f"Returning valid cached LLM response for prompt hash: {cache_key}")
                        all_answers.extend(cached_response["answers"])
                        questions_processed_in_session += len(batch_questions)
                        continue
                    else:
                        logger.warning(f"Cached response for {cache_key} contains non-string answers. Deleting cache and re-generating.")
                        delete_cache(cache_key)
                else:
                    logger.warning(f"Cached response for {cache_key} is not a valid dict/list. Deleting cache and re-generating.")
                    delete_cache(cache_key)

            final_token_estimate = _estimate_tokens(prompt)
            logger.info(f"Final prompt token estimate for batch: {final_token_estimate}")

            if final_token_estimate > self.max_prompt_tokens:
                logger.error(f"FATAL: Prompt size ({final_token_estimate}) exceeds max limit ({self.max_prompt_tokens}). Skipping batch.")
                all_answers.extend(["Error: Prompt too large for LLM."] * len(batch_questions))
                continue

            try:
                response = self._call_gemini_with_retry(prompt)
                logger.info(f"LLM raw response for batch: {response.text!r}")
                parsed_response = self._parse_llm_batch_response(response.text)
                
                if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                    set_cache(cache_key, parsed_response) # Cache the successful response
                    all_answers.extend(parsed_response["answers"])
                    questions_processed_in_session += len(batch_questions)
                else:
                    logger.error(f"LLM response for batch did not contain expected 'answers' list or was malformed: {parsed_response}")
                    all_answers.extend(["Error: LLM did not return answers in the expected format."] * len(batch_questions))

            except Exception as e:
                logger.error(f"Error during LLM API call or response parsing for batch: {e}", exc_info=True)
                all_answers.extend([f"Error: Failed to get response from LLM ({str(e)})." ] * len(batch_questions))
        
        return {"answers": all_answers}

    def _construct_batch_prompt(self, processed_questions: List[Dict[str, Any]]) -> str:
        """
        Constructs a single, detailed prompt for a batch of questions with token-aware context stuffing and Chain-of-Thought reasoning.
        """
        # Use the loaded base prompt template
        base_prompt_template = self.base_prompt_template
        
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
            max_chunks_per_question = 5 # Increased hard limit to allow more context
            chunks_added = 0

            for idx, chunk_obj in enumerate(item['context']):
                if chunks_added >= max_chunks_per_question:
                    logger.info(f"Reached max chunks ({max_chunks_per_question}) for this question. Skipping further chunks.")
                    break

                # Extract content and source information for the prompt
                content = ""
                source_info = ""
                if isinstance(chunk_obj, EmbeddedChunk):
                    content = chunk_obj.chunk.content
                    source_info = f"Source: {chunk_obj.chunk.source_document}"
                    if chunk_obj.chunk.metadata and "page_number" in chunk_obj.chunk.metadata:
                        source_info += f" (Page: {chunk_obj.chunk.metadata['page_number']})"
                    elif chunk_obj.chunk.metadata and "section_title" in chunk_obj.chunk.metadata:
                        source_info += f" (Section: {chunk_obj.chunk.metadata['section_title']})"
                elif isinstance(chunk_obj, Chunk):
                    content = chunk_obj.content
                    source_info = f"Source: {chunk_obj.source_document}"
                    if chunk_obj.metadata and "page_number" in chunk_obj.metadata:
                        source_info += f" (Page: {chunk_obj.metadata['page_number']})"
                    elif chunk_obj.metadata and "section_title" in chunk_obj.metadata:
                        source_info += f" (Section: {chunk_obj.metadata['section_title']})"
                else:
                    logger.warning(f"Unknown chunk object type in _construct_batch_prompt: {type(chunk_obj)}")
                    content = str(chunk_obj)
                    source_info = "Source: Unknown"

                chunk_str = f"[Clause {idx+1}] {source_info} - {content}"

                chunk_tokens = _estimate_tokens(chunk_str)
                if current_tokens + chunk_tokens <= self.max_prompt_tokens:
                    context_for_question.append(chunk_str)
                    current_tokens += chunk_tokens
                    chunks_added += 1
                else:
                    logger.info("Stopping context stuffing for this question to stay within token limit.")
                    break
            
            questions_with_context.append(question_str + "".join(context_for_question))
            logger.debug(f"Context for Question {i+1}:\n{"".join(context_for_question)}")

        if not questions_with_context:
            logger.error("Could not construct any questions within the token limit.")
            return None # Return None to indicate prompt construction failure

        all_questions_block = separator.join(questions_with_context)
        return base_prompt_template.format(questions_block=all_questions_block)

    def _parse_llm_batch_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the LLM's response to extract the structured JSON object with a list of answers.
        Expected format: { "answers": ["Answer 1", "Answer 2", ...] }
        """
        try:
            # Attempt to find JSON block wrapped in ```json ... ```
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If not found, attempt to find a standalone JSON object
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip() # Use group(0) to get the whole matched string
                else:
                    raise ValueError("Could not find a JSON block in the LLM response.")

            decision = json.loads(json_str)
            
            # Validate the structure: expect a dict with an 'answers' key which is a list of strings
            if isinstance(decision, dict) and isinstance(decision.get('answers'), list):
                # Further validate each item in the 'answers' list is a string
                for ans_item in decision['answers']:
                    if not isinstance(ans_item, str):
                        raise json.JSONDecodeError("Each answer in 'answers' list must be a string.", json_str, 0)
                logger.info(f"Successfully parsed {len(decision['answers'])} structured answers from batched response.")
                return decision
            else:
                raise json.JSONDecodeError("Expected a JSON object with an 'answers' key containing a list of strings.", json_str, 0)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid JSON from LLM batch response: {e}. Raw response: {response_text!r}")
            return {"error": "Could not parse LLM response.", "details": response_text}