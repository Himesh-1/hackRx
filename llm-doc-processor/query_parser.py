"""
Query Parser Module
Parses natural language queries and extracts structured information
for better document retrieval and decision making.
"""
import re
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import spacy
from spacy.tokens import Doc
import os
import google.generativeai as genai
from dotenv import load_dotenv
import hashlib # Import hashlib
from utils.cache_utils import get_cache, set_cache # Import cache utilities

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Dataclass to hold the parsed query results."""
    original_query: str
    entities: Dict[str, Any] = field(default_factory=dict)
    intent: str = "general_inquiry"
    keywords: List[str] = field(default_factory=list)
    enhanced_query: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryParser:

    def _clean_query(self, query: str) -> str:
        """
        Cleans the query by converting to lowercase and removing extra whitespace.
        Args:
            query: The input query string.
        Returns:
            The cleaned query string.
        """
        # Simple cleaning: lowercase and strip whitespace
        # More complex cleaning (e.g., removing punctuation) can be added if needed
        return re.sub(r'\s+', ' ', query).strip().lower()

    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parses a single natural language query to extract structured information.
        Args:
            query: The natural language query string.
        Returns:
            A ParsedQuery object containing the extracted information.
        """
        self._load_spacy_model(self.spacy_model) # Ensure model is loaded before use
        cleaned_query = self._clean_query(query)
        doc = self.nlp(cleaned_query) if self.nlp else None
        entities = self._extract_entities(cleaned_query, doc)
        intent = self._determine_intent(cleaned_query)
        keywords = self._extract_keywords(cleaned_query, doc)
        enhanced_query = self._rewrite_queries_batch([query])[0] if hasattr(self, '_rewrite_queries_batch') else query
        confidence = self._calculate_confidence(entities, intent, keywords)
        parsed_query = ParsedQuery(
            original_query=query,
            entities=entities,
            intent=intent,
            keywords=keywords,
            enhanced_query=enhanced_query,
            confidence=confidence,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'query_length': len(query),
                'tokens': [token.text for token in doc] if doc else [],
            }
        )
        return parsed_query

    def _extract_entities(self, query: str, doc: Optional[Doc]) -> Dict[str, Any]:
        """
        Extracts entities from the query using regex patterns and Spacy.
        """
        entities = {}

        # Regex-based entity extraction
        for pattern_name, patterns in [
            ('age', self.age_patterns),
            ('gender', self.gender_patterns),
            ('procedure', self.procedure_patterns),
            ('location', self.location_patterns),
            ('policy_duration', self.policy_duration_patterns),
            ('policy_number', self.policy_number_patterns),
            ('date', self.date_patterns),
            ('money', self.money_patterns),
            ('medical_condition', self.medical_condition_patterns),
            ('insurance_type', self.insurance_type_patterns),
        ]:
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    # For patterns with capturing groups, use the first group
                    if pattern_name in ['procedure', 'money', 'insurance_type']:
                        entities[pattern_name] = match.group(0).strip()
                    elif match.groups():
                        entities[pattern_name] = match.group(1).strip()
                    else:
                        entities[pattern_name] = match.group(0).strip()
                    break # Move to next entity type after first match

        # Spacy-based NER (if doc is available)
        if doc:
            for ent in doc.ents:
                # Prioritize more specific regex matches over general Spacy NER
                if ent.label_ == "GPE" and "location" not in entities: # Geo-political entity (cities, states, countries)
                    entities["location"] = ent.text
                elif ent.label_ == "DATE" and "date" not in entities:
                    entities["date"] = ent.text
                elif ent.label_ == "MONEY" and "money" not in entities:
                    entities["money"] = ent.text
                # Add more Spacy entity types as needed

        return entities

    def _determine_intent(self, query: str) -> str:
        """
        Determines the intent of the query based on predefined patterns.
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return "general_inquiry"

    def _extract_keywords(self, query: str, doc: Optional[Doc]) -> List[str]:
        """
        Extracts keywords from the query using Spacy and predefined lists.
        """
        keywords = set()
        if doc:
            # Add nouns and adjectives as potential keywords
            keywords.update([token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN", "ADJ")])

        # Add keywords from predefined lists if they appear in the query
        for kw_list in [self.insurance_keywords, self.medical_keywords]:
            for keyword in kw_list:
                if keyword in query.lower():
                    keywords.add(keyword)
        return list(keywords)

    

    def _calculate_confidence(self, entities: Dict[str, Any], intent: str, keywords: List[str]) -> float:
        """
        Calculates a confidence score for the parsed query.
        """
        score = 0.0
        if intent != "general_inquiry":
            score += 0.4 # Higher confidence for specific intent
        score += len(entities) * 0.1 # Each extracted entity adds confidence
        score += len(keywords) * 0.05 # Each keyword adds some confidence

        # Cap confidence at 1.0
        return min(score, 1.0)

    def parse_queries(self, queries: List[str]) -> List[ParsedQuery]:
        """
        Batch parses a list of natural language queries to extract structured information.
        Args:
            queries: List of natural language query strings.
        Returns:
            List of ParsedQuery objects.
        """
        return [self.parse_query(q) for q in queries]
    """
    QueryParser class to parse natural language queries and extract structured information.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the QueryParser.
        Args:
            spacy_model: The Spacy model to use for NLP processing.
        """
        self.spacy_model = spacy_model # Store the model name
        self.nlp = None # Initialize lazily
        self._setup_patterns()
        self.gemini_model = None
        self.api_keys = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(',') if key.strip()]
        self.current_hyde_key_index = 0 # Track current key for HyDE
        self.hyde_cache = {}  # Initialize HyDE cache as an instance variable
        if not self.api_keys:
            logger.warning("GEMINI_API_KEYS not found or empty in environment variables. Query rewriting will be disabled.")
        else:
            query_rewrite_model_name = os.getenv("QUERY_REWRITE_MODEL", "gemini-1.5-flash")
            try:
                self.gemini_model = genai.GenerativeModel(query_rewrite_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model '{query_rewrite_model_name}' for HyDE: {e}")

    def _load_spacy_model(self, model_name: str) -> Optional[spacy.Language]:
        """
        Loads the Spacy model, handling potential errors and downloading if necessary.
        Args:
            model_name (str): The name of the Spacy model to load (e.g., "en_core_web_sm").
        Returns:
            Optional[spacy.Language]: The loaded Spacy Language model, or None if loading fails.
        """
        if self.nlp is not None:
            return self.nlp # Model already loaded

        try:
            self.nlp = spacy.load(model_name)
            self.nlp.max_length = 2_000_000  # Increase max_length to handle large texts
            return self.nlp
        except OSError:
            logger.warning(f"Spacy model '{model_name}' not found. Downloading...")
            try:
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
                return self.nlp
            except Exception as e:
                logger.error(f"Failed to download or load Spacy model '{model_name}': {e}")
                return None

    def _setup_patterns(self):
        """
        Initializes regex patterns and keyword mappings for entity extraction.
        These patterns are used to identify specific pieces of information within the user's query.
        """
        # More flexible age patterns
        self.age_patterns = [
            r'(\d{1,3})\s*(?:year|yr|y)s?\s*old',
            r'age\s*is\s*(\d{1,3})',
            r'(\d{1,3})\s*(?:M|F|male|female)',
        ]
        # Expanded gender patterns
        self.gender_patterns = [
            r'\b(male|female|man|woman|boy|girl)\b',
            r'\b(M|F)\b',
        ]
        # Comprehensive medical procedure patterns
        self.procedure_patterns = [
            r'\b((?:knee|hip|heart|cardiac|brain|liver|kidney|lung|spine)\s*(?:surgery|operation|replacement|transplant))\b',
            r'\b(appendectomy|cholecystectomy|bypass|angioplasty|stent|biopsy|dialysis|chemotherapy)\b',
            r'\b(cataract|dental|maternity|orthopedic|neurological)\s*(treatment|procedure|care)',
        ]
        # Location patterns with a focus on Indian cities
        self.location_patterns = [
            r'\b(Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad)\b',
            r'in\s+([A-Z][a-z]+)',
        ]
        # More robust policy duration patterns
        self.policy_duration_patterns = [
            r'(\d+)\s*(month|year)s?\s*old\s*policy',
            r'policy\s*is\s*(\d+)\s*(month|year)s?\s*old',
        ]
        # New: Policy Number patterns
        self.policy_number_patterns = [
            r'\b(?:POL|PLC|PN)\s*[-_]?\s*(\d{6,15})\b', # Common prefixes + digits
            r'\b(\d{6,15})\s*(?:policy|plan)\s*number\b', # Digits followed by policy number
        ]
        # New: Date patterns (flexible)
        self.date_patterns = [
            r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b', # DD-MM-YYYY, DD/MM/YY, etc.
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b', # Month Day, Year
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', # Day Month Year
        ]
        # New: Monetary Value patterns
        self.money_patterns = [
            r'(?:Rs\. Re?|INR|₹)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?',
            r'(?:\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:lakh|crore)\s*rupees\b',
            r'(?:\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:lakh|crore)\b',
        ]
        # New: Medical Condition/Disease patterns
        self.medical_condition_patterns = [
            r'\b(diabetes|hypertension|cancer|heart\s*disease|kidney\s*failure|asthma|arthritis|thyroid)\b',
            r'\b(fever|cold|flu|infection|allergy|migraine)\b',
            r'\b(pre-existing\s*condition)\b',
        ]
        # New: Insurance Type patterns
        self.insurance_type_patterns = [
            r'\b((?:mediclaim|health|life|travel|motor|home)\s*(?:insurance|policy|plan))\b',
        ]

        # Expanded insurance and medical keywords
        self.insurance_keywords = [
            'claim', 'coverage', 'policy', 'premium', 'deductible', 'reimbursement',
            'pre-existing', 'waiting period', 'cashless', 'sum assured', 'hospitalization',
            'renewal', 'endorsement', 'nominee', 'insurer', 'insured', 'grace period',
            'sub-limit', 'co-pay', 'network hospital', 'portability', 'free-look period',
            'critical illness', 'personal accident', 'daily cash', 'domiciliary hospitalization',
            'ayush treatment', 'organ donor', 'no claim bonus', 'maternity benefit',
            'out-patient department', 'opd', 'annual health check-up', 'wellness program'
        ]
        self.medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'doctor', 'hospital', 'emergency',
            'surgery', 'illness', 'disease', 'injury', 'symptoms', 'therapy', 'consultation',
            'prescription', 'medical report', 'discharge summary', 'ambulance', 'icu', 'room rent',
            'operation theatre', 'pathology', 'radiology', 'physiotherapy', 'rehabilitation',
            'vaccination', 'preventive care', 'chronic disease', 'acute illness', 'congenital disease'
        ]
        # More specific intent patterns
        self.intent_patterns = {
            'coverage_inquiry': [r'\b(cover|coverage|covered|include|included|eligible|reimburse)\b'],
            'waiting_period_inquiry': [r'\b(waiting\s*period|wait\s*period|how\s*long\s*before)\b'],
            'claim_process_inquiry': [r'\b(claim\s*process|how\s*to\s*claim|file\s*a\s*claim)\b'],
            'cost_inquiry': [r'\b(cost|price|amount|expense|charge|premium)\b'],
            'definition_inquiry': [r'\b(define|definition|meaning|what\s*is)\b'],
            'policy_details_inquiry': [r'\b(policy\s*details|plan\s*details|terms\s*and\s*conditions|document)\b'],
            'hospital_network_inquiry': [r'\b(network\s*hospital|empanelled\s*hospital|list\s*of\s*hospitals)\b'],
            'pre_existing_inquiry': [r'\b(pre-existing\s*disease|ped)\b'],
            'maternity_inquiry': [r'\b(maternity|pregnancy|childbirth)\b'],
            'health_checkup_inquiry': [r'\b(health\s*check-up|wellness\s*program)\b'],
            'renewal_inquiry': [r'\b(renewal|renew|expire)\b'],
            'portability_inquiry': [r'\b(portability|port\s*policy)\b'],
            'general_inquiry': [r'\b(what|how|when|where|is|are|can)\b'] # Broad catch-all
        }


    def _call_gemini_for_hyde_with_retry(self, prompt: str, max_retries: int = int(os.getenv("HYDE_MAX_RETRIES", 3)), initial_delay: float = 0.5, timeout: float = 30.0) -> str:
        """
        Calls the Gemini API for HyDE generation with retry logic, exponential backoff, key rotation, and robust error handling.
        Returns a fallback answer if all keys are exhausted or on repeated failure.
        """
        import threading
        query_rewrite_model_name = os.getenv("QUERY_REWRITE_MODEL", "gemini-1.5-flash")
        last_exception = None
        for i in range(max_retries * len(self.api_keys)):
            try:
                genai.configure(api_key=self.api_keys[self.current_hyde_key_index])
                self.gemini_model = genai.GenerativeModel(query_rewrite_model_name)
                logger.info(f"Attempt {i+1}/{max_retries*len(self.api_keys)} with HyDE key index {self.current_hyde_key_index} (Key: {self.api_keys[self.current_hyde_key_index][:5]}...) to call Gemini for HyDE.")
                # Add timeout to Gemini call
                result = [None]
                def call_model():
                    try:
                        response = self.gemini_model.generate_content(prompt)
                        result[0] = response.text.strip()
                    except Exception as e:
                        result.append(e)
                t = threading.Thread(target=call_model)
                t.start()
                t.join(timeout)
                if t.is_alive():
                    logger.error(f"HyDE Gemini call timed out after {timeout} seconds.")
                    t.join(0) # Let thread die
                    raise TimeoutError(f"Gemini call timed out after {timeout} seconds.")
                if isinstance(result[0], Exception):
                    raise result[0]
                if result[0] is not None:
                    return result[0]
                else:
                    raise Exception("Unknown error in Gemini HyDE call.")
            except Exception as e:
                last_exception = e
                logger.error(f"HyDE generation failed (Attempt {i+1}/{max_retries*len(self.api_keys)}): {e}")
                if "rate limit" in str(e).lower() or "resource exhausted" in str(e).lower() or "quota" in str(e).lower() or isinstance(e, TimeoutError):
                    logger.warning("HyDE rate/timeout/quota limit hit. Attempting to switch API key.")
                    self.current_hyde_key_index = (self.current_hyde_key_index + 1) % len(self.api_keys)
                    logger.warning(f"Switched to HyDE key index: {self.current_hyde_key_index}")
                    time.sleep(initial_delay * (2 ** (i % max_retries))) # Exponential backoff per key
                else:
                    # For other errors, log and break
                    logger.error(f"Non-recoverable error in HyDE: {e}")
                    break
        logger.error(f"Failed to generate hypothetical answer after {max_retries*len(self.api_keys)} attempts across all keys. Last error: {last_exception}")
        # Return a fallback answer to avoid crashing
        return "[HyDE unavailable: fallback answer used]"

    def _parse_hyde_batch_response(self, response_text: str, num_queries: int) -> List[str]:
        """
        Parses the LLM's batched response for hypothetical answers.
        Expected format: JSON object with "hypothetical_answers" as a list of strings.
        """
        try:
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response_text.strip() # Try to parse directly if no markdown block

            parsed_data = json.loads(json_str)
            
            if isinstance(parsed_data, dict) and isinstance(parsed_data.get("hypothetical_answers"), list):
                if len(parsed_data["hypothetical_answers"]) == num_queries:
                    logger.info(f"Successfully parsed {len(parsed_data['hypothetical_answers'])} hypothetical answers.")
                    return parsed_data["hypothetical_answers"]
                else:
                    logger.warning(f"Mismatch in number of hypothetical answers. Expected {num_queries}, got {len(parsed_data['hypothetical_answers'])}. Raw: {response_text!r}")
                    return ["" for _ in range(num_queries)] # Return empty strings for all
            else:
                raise ValueError("Expected 'hypothetical_answers' list in LLM response.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse HyDE batch response: {e}. Raw response: {response_text!r}")
            return ["" for _ in range(num_queries)] # Return empty strings on parsing failure

    def _rewrite_queries_batch(self, queries: List[str]) -> List[str]:
        """
        Generates hypothetical answers for a batch of user queries using HyDE.
        """
        if not self.gemini_model or not self.api_keys:
            logger.warning("Gemini model not initialized or no API keys available, skipping HyDE generation.")
            return queries # Fallback to original queries

        # Check cache for each query
        cached_hydes = [get_cache(f"hyde_query_{hashlib.md5(q.encode('utf-8')).hexdigest()}") for q in queries]
        
        queries_to_process = []
        indices_to_fill = []
        final_enhanced_queries = ["" for _ in queries]

        for i, cached_hyde in enumerate(cached_hydes):
            if cached_hyde:
                logger.info(f"Using cached hypothetical answer for query: '{queries[i]}'")
                final_enhanced_queries[i] = cached_hyde
            else:
                queries_to_process.append(queries[i])
                indices_to_fill.append(i)

        if not queries_to_process:
            return final_enhanced_queries

        prompt = f"""Given the user queries, generate a hypothetical, but plausible, answer for each that could be found in a relevant document. Return a JSON object with a single key 'hypothetical_answers' which is a list of strings. Each string in the list should be a concise and direct hypothetical answer for the corresponding query. Do not include any conversational filler or introductions. Just the JSON.

User Queries: {json.dumps(queries_to_process)}

Hypothetical Answers (JSON):"""

        import time
        start_time = time.time()
        try:
            hyde_answers_text = self._call_gemini_for_hyde_with_retry(prompt)
            hyde_answers = self._parse_hyde_batch_response(hyde_answers_text, len(queries_to_process))

            for i, hyde_answer in enumerate(hyde_answers):
                original_query_index = indices_to_fill[i]
                enhanced_query = f"{queries[original_query_index]} {hyde_answer}"
                set_cache(f"hyde_query_{hashlib.md5(queries[original_query_index].encode('utf-8')).hexdigest()}", enhanced_query)
                final_enhanced_queries[original_query_index] = enhanced_query

        except Exception as e:
            logger.error(f"Failed to generate hypothetical answers for batch: {e}")
            # Fallback for failed queries
            for i in indices_to_fill:
                final_enhanced_queries[i] = queries[i] + " [HyDE unavailable: fallback used]"

        elapsed = time.time() - start_time
        logger.info(f"HyDE batch processing time: {elapsed:.2f} seconds for {len(queries_to_process)} queries.")
        return final_enhanced_queries

        
            
