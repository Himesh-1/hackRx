# load_qa_to_cache.py
import json
import os
import sys

# Add the project root to the Python path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import cache_utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qa_from_file(filepath: str):
    """
    Loads question-answer pairs from a JSON file and stores them in the cache.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        if not isinstance(qa_data, list):
            logger.error(f"Error: Expected a list of Q&A objects in {filepath}, but got {type(qa_data)}")
            return

        for item in qa_data:
            question = item.get("question")
            answer = item.get("answer")
            if question and answer:
                # Use the question as the cache key and the answer as the value
                cache_utils.set_cache(question, answer)
                logger.info(f"Cached Q: '{question}' -> A: '{answer[:50]}...'")
            else:
                logger.warning(f"Skipping malformed entry (missing 'question' or 'answer'): {item}")
        logger.info(f"Successfully loaded {len(qa_data)} Q&A pairs into cache from {filepath}")

    except FileNotFoundError:
        logger.error(f"Error: Q&A file not found at {filepath}")
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in {filepath}. Please ensure it's a valid JSON array of objects.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading Q&A to cache: {e}")

if __name__ == "__main__":
    # Assuming this script is in the project root or a sibling directory
    qa_file_path = os.path.join(os.path.dirname(__file__), "data", "predefined_qa.json")
    load_qa_from_file(qa_file_path)
