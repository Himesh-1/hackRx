"""
Query Parser Module
Parses natural language queries and extracts structured information
for better document retrieval and decision making.
"""
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import spacy
from spacy.tokens import Doc
import os
import google.generativeai as genai
from dotenv import load_dotenv

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
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. Query rewriting will be disabled.")
        else:
            genai.configure(api_key=self.api_key)
            query_rewrite_model_name = os.getenv("QUERY_REWRITE_MODEL", "gemini-1.5-flash")
            try:
                self.gemini_model = genai.GenerativeModel(query_rewrite_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model '{query_rewrite_model_name}': {e}")

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
            r'\b(knee|hip|heart|cardiac|brain|liver|kidney|lung|spine)\s*(surgery|operation|replacement|transplant)',
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
        # Expanded insurance and medical keywords
        self.insurance_keywords = [
            'claim', 'coverage', 'policy', 'premium', 'deductible', 'reimbursement',
            'pre-existing', 'waiting period', 'cashless', 'sum assured', 'hospitalization',
        ]
        self.medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'doctor', 'hospital', 'emergency',
            'surgery', 'illness', 'disease', 'injury',
        ]
        # More specific intent patterns
        self.intent_patterns = {
            'claim_inquiry': [r'\b(claim|coverage|covered|eligible|reimburse)\b'],
            'policy_check': [r'\b(policy|insurance|plan)\s*(details|status|terms)\b'],
            'cost_inquiry': [r'\b(cost|price|amount|expense)\b'],
            'eligibility_check': [r'\b(eligible|qualify|can\s+I\s+get)\b'],
        }

    def _rewrite_query(self, query: str) -> str:
        """
        Rewrites the user query into a more detailed and specific question using an LLM.
        This helps in retrieving more relevant documents from the knowledge base.
        Args:
            query (str): The original user query.
        Returns:
            str: The rewritten, more specific query, or the original query if rewriting fails.
        """
        logger.info(f"Skipping LLM-based query rewriting. Returning original query: '{query}'")
        return query

    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parses a natural language query to extract structured information. This method
        is designed to handle vague queries by cleaning the input, extracting entities,
        determining intent, and generating an enhanced query for better retrieval.
        Args:
            query: The natural language query string.
        Returns:
            A ParsedQuery object containing the extracted information.
        """
        logger.info(f"Starting query parsing for: '{query}'")
        self._load_spacy_model(self.spacy_model) # Ensure model is loaded before use
        cleaned_query = self._clean_query(query)
        doc = self.nlp(cleaned_query) if self.nlp else None
        entities = self._extract_entities(cleaned_query, doc)
        intent = self._determine_intent(cleaned_query)
        keywords = self._extract_keywords(cleaned_query, doc)
        # The main enhancement is now the LLM-rewritten query
        enhanced_query = self._rewrite_query(query)
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
        logger.info(f"Query parsed with confidence {confidence:.2f}. Enhanced query: '{enhanced_query}'")
        return parsed_query

    def _clean_query(self, query: str) -> str:
        """Cleans and normalizes the query text."""
        return query.lower().strip()

    def _extract_entities(self, query: str, doc: Optional[Doc]) -> Dict[str, Any]:
        """Extracts structured entities from the query using regex and Spacy."""
        entities = {}
        if doc:
            # Use Spacy for NER
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'DATE', 'MONEY']:
                    entities[ent.label_.lower()] = ent.text

        # Use regex for more specific patterns
        entities.update({
            'age': self._extract_pattern(query, self.age_patterns),
            'gender': self._extract_pattern(query, self.gender_patterns),
            'procedure': self._extract_pattern(query, self.procedure_patterns),
            'location': self._extract_pattern(query, self.location_patterns),
            'policy_duration': self._extract_policy_duration(query),
        })

        # Filter out None values
        return {k: v for k, v in entities.items() if v}

    def _extract_pattern(self, query: str, patterns: List[str]) -> Optional[str]:
        """Extracts the first matching pattern from the query."""
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None

    def _extract_policy_duration(self, query: str) -> Optional[Dict[str, Any]]:
        """Extracts policy duration with units."""
        for pattern in self.policy_duration_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return {'value': int(match.group(1)), 'unit': match.group(2)}
        return None

    def _determine_intent(self, query: str) -> str:
        """Determines the query's intent based on predefined patterns."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return 'general_inquiry'

    def _extract_keywords(self, query: str, doc: Optional[Doc]) -> List[str]:
        """Extracts relevant keywords using NLP and predefined lists."""
        keywords = set()
        if doc:
            # Extract nouns, verbs, and adjectives as keywords
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                    keywords.add(token.lemma_)
        # Add keywords from predefined lists
        for keyword in self.insurance_keywords + self.medical_keywords:
            if keyword in query.lower():
                keywords.add(keyword)
        return list(keywords)

    def _calculate_confidence(self, entities: Dict[str, Any], intent: str, keywords: List[str]) -> float:
        """Calculates a confidence score based on the extracted information."""
        score = 0.0
        if entities:
            score += 0.5
        if intent != 'general_inquiry':
            score += 0.3
        if keywords:
            score += 0.2
        return min(score, 1.0)


# Example usage for testing
if __name__ == "__main__":
    # Make sure to have a .env file with GEMINI_API_KEY for this to work
    parser = QueryParser()
    test_queries = [
        "I am a 45-year-old man and I need to know if my knee surgery is covered by my 2-year-old policy.",
        "Vague query about heart problems.",
        "Is my policy still valid?",
        "What is the claim process for maternity treatment in Mumbai?",
    ]
    for q in test_queries:
        parsed_result = parser.parse_query(q)
        print(f"--- Original Query: '{q}' ---")
        print(f"   Intent: {parsed_result.intent}")
        print(f"   Entities: {parsed_result.entities}")
        print(f"   Keywords: {parsed_result.keywords}")
        print(f"   Enhanced Query: {parsed_result.enhanced_query}")
        print(f"   Confidence: {parsed_result.confidence:.2f}\n")