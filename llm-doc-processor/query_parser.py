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
        enhanced_query = self._rewrite_query(query) if hasattr(self, '_rewrite_query') else query
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

    def _rewrite_query(self, query: str) -> str:
        """
        Rewrites the query using HyDE (Hypothetical Document Embedding) if enabled.
        This involves generating a hypothetical answer and using it to enhance the query.
        """
        if not self.gemini_model or not self.api_keys:
            logger.warning("Gemini model not initialized or no API keys available, skipping HyDE generation.")
            return query

        # Check cache first
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        cached_hyde = get_cache(f"hyde_query_{query_hash}")
        if cached_hyde:
            logger.info(f"Using cached hypothetical answer for query: {query}")
            return cached_hyde

        prompt = f"""Given the user query, generate a hypothetical, but plausible, answer that could be found in a relevant document. This hypothetical answer should be concise and directly address the query. Do not include any conversational filler or introductions. Just the hypothetical answer.

User Query: {query}

Hypothetical Answer:"""

        try:
            # Rotate API key proactively if interval is met
            self.hyde_queries_processed_in_session += 1
            if self.hyde_queries_processed_in_session >= self.hyde_key_switch_interval:
                self.current_hyde_key_index = (self.current_hyde_key_index + 1) % len(self.api_keys)
                self.hyde_queries_processed_in_session = 0 # Reset counter
                logger.info(f"Proactively switched to HyDE key index: {self.current_hyde_key_index}")

            hyde_answer = self._call_gemini_for_hyde_with_retry(prompt)
            enhanced_query = f"{query} {hyde_answer}"
            set_cache(f"hyde_query_{query_hash}", enhanced_query) # Cache the enhanced query
            return enhanced_query
        except Exception as e:
            logger.error(f"Failed to generate hypothetical answer for query '{query}': {e}")
            return query # Fallback to original query on failure

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
        self.hyde_queries_processed_in_session = 0 # New: Track queries processed for proactive key rotation
        self.hyde_key_switch_interval = int(os.getenv("HYDE_KEY_SWITCH_INTERVAL", 5)) # New: Configurable interval
        self.hyde_cache = {}  # Initialize HyDE cache as an instance variable
        if not self.api_keys:
            logger.warning("GEMINI_API_KEYS not found or empty in environment variables. Query rewriting will be disabled.")
        else:
            query_rewrite_model_name = os.getenv("QUERY_REWRITE_MODEL", "gemini-1.5-flash")
            try:
                self.gemini_model = genai.GenerativeModel(query_rewrite_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model '{query_rewrite_model_name}' for HyDE: {e}")

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
        self.hyde_queries_processed_in_session = 0 # New: Track queries processed for proactive key rotation
        self.hyde_key_switch_interval = int(os.getenv("HYDE_KEY_SWITCH_INTERVAL", 5)) # New: Configurable interval
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


    def _call_gemini_for_hyde_with_retry(self, prompt: str, max_retries: int = int(os.getenv("HYDE_MAX_RETRIES", 3)), initial_delay: float = 0.5) -> str:
        """
        Calls the Gemini API for HyDE generation with retry logic and exponential backoff, including key rotation.
        """
        for i in range(max_retries):
            try:
                genai.configure(api_key=self.api_keys[self.current_hyde_key_index])
                logger.info(f"Attempt {i+1}/{max_retries} with HyDE key index {self.current_hyde_key_index} to call Gemini for HyDE.")
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"HyDE generation failed (Attempt {i+1}/{max_retries}): {e}")
                if "rate limit" in str(e).lower() or "resource exhausted" in str(e).lower() or "quota" in str(e).lower():
                    logger.warning("HyDE rate limit hit. Attempting to switch API key.")
                    self.current_hyde_key_index = (self.current_hyde_key_index + 1) % len(self.api_keys)
                    logger.warning(f"Switched to HyDE key index: {self.current_hyde_key_index}")
                    time.sleep(initial_delay * (2 ** i)) # Exponential backoff
                else:
                    raise # Re-raise other exceptions
        raise Exception(f"Failed to generate hypothetical answer after {max_retries} retries across all keys.")

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
        for i, cached_hyde in enumerate(cached_hydes):
            if cached_hyde:
                logger.info(f"Using cached hypothetical answer for query: '{queries[i]}'")
            else:
                queries_to_process.append(queries[i])
                indices_to_fill.append(i)

        
            